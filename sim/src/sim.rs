//! The simulation loop (physics at 1 kHz, IMU at 200 Hz, stereo camera at
//! 20 Hz) and the dataset writer, in the EuRoC ASL layout so that the same
//! odometry runs on simulated and real data. Besides EuRoC's folders it
//! writes, for the left camera, the true depth (16-bit PNG, millimeters) and
//! the kind of surface of each pixel (floor, wall, ceiling, obstacle).

use crate::math::{v3, M3, V3};
use crate::png;
use crate::rover::{render, Camera, ImuSample, Rover, State, NOISE};
use crate::world::World;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const PHYSICS_DT: f64 = 0.001;
pub const IMU_EVERY: usize = 5; // 200 Hz
pub const CAMERA_EVERY: usize = 50; // 20 Hz
const T0_NS: u64 = 1_000_000_000;

#[derive(Serialize)]
pub struct FrameOut {
    pub t: f64,
    pub left: String,
    pub right: String,
}

pub struct Sim {
    pub world: World,
    pub rover: Rover,
    tick: usize,
    hist: [State; 2], // the two physics states before the current one
    out: Option<Writer>,
    seed: u64,
}

struct Writer {
    root: PathBuf,
    cam: [BufWriter<File>; 2],
    imu: BufWriter<File>,
    gt: BufWriter<File>,
}

fn ns(t: f64) -> u64 {
    T0_NS + (t * 1e9).round() as u64
}

fn yaml_camera(cam: &Camera) -> String {
    let r = &cam.r_bc.0;
    let t = cam.t_bc;
    format!(
        "sensor_type: camera\ncomment: simulated rover camera\nT_BS:\n  cols: 4\n  rows: 4\n  data: [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 0.0, 0.0, 0.0, 1.0]\nrate_hz: 20\nresolution: [{}, {}]\ncamera_model: pinhole\nintrinsics: [{}, {}, {}, {}]\ndistortion_model: radial-tangential\ndistortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n",
        r[0][0], r[0][1], r[0][2], t.x, r[1][0], r[1][1], r[1][2], t.y, r[2][0], r[2][1], r[2][2], t.z, cam.w, cam.h, cam.fx, cam.fy, cam.cx, cam.cy
    )
}

impl Writer {
    fn create(root: &Path, rover: &Rover) -> std::io::Result<Writer> {
        let mav = root.join("mav0");
        for d in ["cam0/data", "cam1/data", "imu0", "state_groundtruth_estimate0", "depth0/data", "label0/data"] {
            fs::create_dir_all(mav.join(d))?;
        }
        for (k, cam) in rover.cams.iter().enumerate() {
            fs::write(mav.join(format!("cam{}/sensor.yaml", k)), yaml_camera(cam))?;
        }
        fs::write(
            mav.join("imu0/sensor.yaml"),
            format!(
                "sensor_type: imu\ncomment: simulated, the noise model of EuRoC's ADIS16448\nT_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\nrate_hz: 200\ngyroscope_noise_density: {}\ngyroscope_random_walk: {}\naccelerometer_noise_density: {}\naccelerometer_random_walk: {}\n",
                NOISE.gyro_noise, NOISE.gyro_walk, NOISE.acc_noise, NOISE.acc_walk
            ),
        )?;
        let open = |p: PathBuf, header: &str| -> std::io::Result<BufWriter<File>> {
            let mut f = BufWriter::new(File::create(p)?);
            writeln!(f, "{}", header)?;
            Ok(f)
        };
        Ok(Writer {
            root: root.to_path_buf(),
            cam: [open(mav.join("cam0/data.csv"), "#timestamp [ns],filename")?, open(mav.join("cam1/data.csv"), "#timestamp [ns],filename")?],
            imu: open(mav.join("imu0/data.csv"), "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")?,
            gt: open(
                mav.join("state_groundtruth_estimate0/data.csv"),
                "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]",
            )?,
        })
    }
}

impl Sim {
    pub fn new(world: World, rover: Rover, out: Option<&Path>, seed: u64) -> std::io::Result<Sim> {
        let out = match out {
            Some(p) => Some(Writer::create(p, &rover)?),
            None => None,
        };
        let s = rover.s;
        Ok(Sim { world, rover, tick: 0, hist: [s, s], out, seed })
    }

    pub fn camera(&self) -> &Camera {
        &self.rover.cams[0]
    }

    /// Advance by one camera period with the given command; returns the IMU
    /// samples and the stereo frame taken at the end.
    pub fn step(&mut self, v_cmd: f64, w_cmd: f64) -> std::io::Result<(Vec<ImuSample>, Option<FrameOut>)> {
        let mut imu = Vec::new();
        let mut frame = None;
        for _ in 0..CAMERA_EVERY {
            self.rover.integrate(PHYSICS_DT, v_cmd, w_cmd, &self.world);
            self.tick += 1;
            let [a, b] = self.hist;
            let c = self.rover.s;
            // the sample at b uses a, b, c (b is one physics step behind)
            if self.tick % IMU_EVERY == 1 && self.tick > 2 {
                let s = self.rover.imu(&a, &b, &c, PHYSICS_DT * IMU_EVERY as f64);
                self.write_imu(&s, &b)?;
                imu.push(s);
            }
            self.hist = [b, c];
            if self.tick % CAMERA_EVERY == 0 {
                frame = Some(self.write_frame()?);
            }
        }
        Ok((imu, frame))
    }

    fn write_imu(&mut self, s: &ImuSample, st: &State) -> std::io::Result<()> {
        let (r, p) = self.rover.pose(st);
        let bg = self.rover.bg;
        let ba = self.rover.ba;
        if let Some(w) = &mut self.out {
            writeln!(w.imu, "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}", ns(s.t), s.gyro.x, s.gyro.y, s.gyro.z, s.acc.x, s.acc.y, s.acc.z)?;
            let q = r.quat();
            let v = v3(st.v * st.yaw.cos(), st.v * st.yaw.sin(), 0.0);
            writeln!(
                w.gt,
                "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                ns(st.t),
                p.x,
                p.y,
                p.z,
                q[0],
                q[1],
                q[2],
                q[3],
                v.x,
                v.y,
                v.z,
                bg.x,
                bg.y,
                bg.z,
                ba.x,
                ba.y,
                ba.z
            )?;
        }
        Ok(())
    }

    fn write_frame(&mut self) -> std::io::Result<FrameOut> {
        let t = self.rover.s.t;
        let (r, p) = self.rover.pose(&self.rover.s);
        let seed = self.seed.wrapping_add(self.tick as u64);
        let left = render(&self.world, &self.rover.cams[0], &r, p, seed);
        let right = render(&self.world, &self.rover.cams[1], &r, p, seed ^ 0xABCD);
        let name = format!("{}.png", ns(t));
        let mut paths = [String::new(), String::new()];
        if let Some(w) = &mut self.out {
            let mav = w.root.join("mav0");
            let cam = &self.rover.cams[0];
            for (k, img) in [&left, &right].iter().enumerate() {
                let path = mav.join(format!("cam{}/data/{}", k, name));
                fs::write(&path, png::encode_gray(cam.w, cam.h, 8, &img.gray))?;
                writeln!(w.cam[k], "{},{}", ns(t), name)?;
                paths[k] = path.to_string_lossy().into_owned();
            }
            let depth: Vec<u8> = left.depth_mm.iter().flat_map(|d| d.to_be_bytes()).collect();
            fs::write(mav.join(format!("depth0/data/{}", name)), png::encode_gray(cam.w, cam.h, 16, &depth))?;
            fs::write(mav.join(format!("label0/data/{}", name)), png::encode_gray(cam.w, cam.h, 8, &left.label))?;
            for f in w.cam.iter_mut() {
                f.flush()?;
            }
            w.imu.flush()?;
            w.gt.flush()?;
        }
        let [l, r] = paths;
        Ok(FrameOut { t, left: l, right: r })
    }

    pub fn rotation(&self) -> M3 {
        self.rover.pose(&self.rover.s).0
    }
    pub fn position(&self) -> V3 {
        self.rover.pose(&self.rover.s).1
    }
}
