#!/bin/sh
# Run the stereo-inertial odometry with loop closure on every EuRoC sequence,
# four at a time; one JSON result, one trajectory and the closed loops per sequence.
cd "$(dirname "$0")/.."
EXE=vio/build/rover-vio; [ -f vio/build/rover-vio.exe ] && EXE=vio/build/rover-vio.exe
mkdir -p results/vio results/traj results/loops
ls data/euroc | grep -E '^(MH|V1|V2)_' | xargs -P ${JOBS:-4} -I{} sh -c "$EXE data/euroc/{} --voc data/voc/rover.voc --out results/traj/{}.txt --json results/vio/{}.json --loops results/loops/{}.txt > /dev/null 2> results/vio/{}.log"
