//! rover-sim: the simulated rover.
//!
//!   rover-sim generate OUT_DIR [--seed N] [--obstacles N] [--laps N]
//!       drive two laps around the room (a pure-pursuit controller on a
//!       rectangle 1.5 m from the walls) and write the dataset
//!   rover-sim serve
//!       a step server on stdin/stdout, one JSON object per line:
//!       {"op": "reset", "seed": 1, "obstacles": 6, "out": "run/sim", "x": 1.5, "y": 1.5, "yaw": 0}
//!       {"op": "step", "v": 0.5, "w": 0.1}      one camera period (50 ms)
//!       {"op": "close"}

mod math;
mod png;
mod rover;
mod sim;
mod world;

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;

fn pure_pursuit(x: f64, y: f64, yaw: f64, path: &[(f64, f64)], target: &mut usize, lookahead: f64) -> (f64, f64) {
    // advance the target along the path while it is closer than the lookahead
    let n = path.len();
    while ((path[*target % n].0 - x).powi(2) + (path[*target % n].1 - y).powi(2)).sqrt() < lookahead {
        *target += 1;
    }
    let (tx, ty) = path[*target % n];
    let angle = (ty - y).atan2(tx - x) - yaw;
    let a = angle.sin().atan2(angle.cos());
    let v = 0.6 * (1.0 - (a.abs() / 1.5).min(0.8));
    let w = 2.0 * v * a.sin() / lookahead;
    (v, w.clamp(-1.2, 1.2))
}

fn generate(out: &str, seed: u64, obstacles: usize, laps: usize) -> std::io::Result<()> {
    let world = world::World::random(seed, obstacles);
    let rover = rover::Rover::new(1.5, 1.5, 0.0, seed);
    let mut s = sim::Sim::new(world, rover, Some(Path::new(out)), seed)?;
    // a dense rectangle path, laps times around
    let corners = [(1.5, 1.5), (10.5, 1.5), (10.5, 8.5), (1.5, 8.5)];
    let mut path = Vec::new();
    for k in 0..4 {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        for i in 0..50 {
            let u = i as f64 / 50.0;
            path.push((a.0 + u * (b.0 - a.0), a.1 + u * (b.1 - a.1)));
        }
    }
    let mut target = 1;
    // stand still for two seconds (the odometry estimates gravity at the start)
    for _ in 0..40 {
        s.step(0.0, 0.0)?;
    }
    let total = path.len() * laps;
    let mut frames = 0;
    while target < total && frames < 20 * 600 {
        let st = s.rover.s;
        let (v, w) = pure_pursuit(st.x, st.y, st.yaw, &path, &mut target, 0.8);
        s.step(v, w)?;
        frames += 1;
    }
    eprintln!("generated {} frames, collided: {}", frames + 40, s.rover.collided);
    Ok(())
}

fn serve() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut sim: Option<sim::Sim> = None;
    for line in stdin.lock().lines() {
        let line = line?;
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                writeln!(out, "{}", json!({"error": e.to_string()}))?;
                continue;
            }
        };
        let reply = match req["op"].as_str().unwrap_or("") {
            "reset" => {
                let seed = req["seed"].as_u64().unwrap_or(1);
                let world = world::World::random(seed, req["obstacles"].as_u64().unwrap_or(6) as usize);
                let rover = rover::Rover::new(req["x"].as_f64().unwrap_or(1.5), req["y"].as_f64().unwrap_or(1.5), req["yaw"].as_f64().unwrap_or(0.0), seed);
                let dir = req["out"].as_str().map(|s| s.to_string());
                let s = sim::Sim::new(world, rover, dir.as_deref().map(Path::new), seed)?;
                let c = s.camera();
                let r = json!({
                    "size": s.world.size, "boxes": s.world.boxes, "radius": s.rover.radius,
                    "camera": {"w": c.w, "h": c.h, "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy},
                    "limits": {"v": s.rover.max_v, "w": s.rover.max_w, "acc": s.rover.max_acc, "tau": s.rover.tau},
                    "dt": sim::PHYSICS_DT * sim::CAMERA_EVERY as f64
                });
                sim = Some(s);
                r
            }
            "step" => match sim.as_mut() {
                None => json!({"error": "reset first"}),
                Some(s) => {
                    let (imu, frame) = s.step(req["v"].as_f64().unwrap_or(0.0), req["w"].as_f64().unwrap_or(0.0))?;
                    let st = s.rover.s;
                    let p = s.position();
                    json!({
                        "t": st.t,
                        "truth": {"x": st.x, "y": st.y, "z": p.z, "yaw": st.yaw, "v": st.v, "w": st.w},
                        "collided": s.rover.collided,
                        "frame": frame,
                        "imu": imu.iter().map(|m| [m.t, m.gyro.x, m.gyro.y, m.gyro.z, m.acc.x, m.acc.y, m.acc.z]).collect::<Vec<_>>(),
                    })
                }
            },
            "close" => {
                writeln!(out, "{}", json!({"closed": true}))?;
                return Ok(());
            }
            other => json!({"error": format!("unknown op {}", other)}),
        };
        writeln!(out, "{}", reply)?;
        out.flush()?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, d: u64| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d);
    let r = match args.get(1).map(|s| s.as_str()) {
        Some("generate") if args.len() > 2 => generate(&args[2], get("--seed", 1), get("--obstacles", 6) as usize, get("--laps", 2) as usize),
        Some("serve") => serve(),
        _ => {
            eprintln!("usage: rover-sim generate OUT_DIR [--seed N] [--obstacles N] [--laps N] | rover-sim serve");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("rover-sim: {}", e);
        std::process::exit(1);
    }
}
