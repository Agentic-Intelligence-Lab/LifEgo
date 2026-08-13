# Revo2 tune_hand (6-finger mode) - Right hand
# Keys: 0=ThumbAux 1=Thumb 2=Index 3=Middle 4=Ring 5=Pinky
import asyncio, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainco-hand-sdk", "python"))
from common_imports import logger, libstark, modbus_open

PORT = "/dev/ttyUSB0"
BAUD = libstark.Baudrate.Baud460800
SID = 0x7F
HW = libstark.StarkHardwareType.Revo2Touch

FINGERS = [
    ("0", "ThumbAux", 0, libstark.FingerId.Thumb),
    ("1", "Thumb",    1, libstark.FingerId.ThumbAux),
    ("2", "Index",    2, libstark.FingerId.Index),
    ("3", "Middle",   3, libstark.FingerId.Middle),
    ("4", "Ring",     4, libstark.FingerId.Ring),
    ("5", "Pinky",    5, libstark.FingerId.Pinky),
]

CTRL = {
    "0": ("ThumbAux", libstark.FingerId.Thumb),
    "1": ("Thumb",    libstark.FingerId.ThumbAux),
    "2": ("Index",    libstark.FingerId.Index),
    "3": ("Middle",   libstark.FingerId.Middle),
    "4": ("Ring",     libstark.FingerId.Ring),
    "5": ("Pinky",    libstark.FingerId.Pinky),
}

poses = {}

async def show(client):
    status = await client.get_motor_status(SID)
    print("\n" + "-" * 55)
    print(f"{'Finger':>10} {'Pos':>6} {'Speed':>6} {'Current':>6}")
    print("-" * 55)
    for _, name, idx, _ in FINGERS:
        print(f"{name:>10} {status.positions[idx]:>6} {status.speeds[idx]:>6} {status.currents[idx]:>6}")
    print("-" * 55)
    return [status.positions[i] for i in range(6)]

async def set_one(client, fid, pos, dur=300):
    await client.set_finger_position_with_millis(SID, fid, pos, dur)
    await asyncio.sleep(dur / 1000.0 + 0.1)

async def set_list(client, positions, dur=300):
    await client.set_finger_positions_and_durations(SID, positions, [dur] * 6)
    await asyncio.sleep(dur / 1000.0 + 0.1)

async def main():
    client = await libstark.modbus_open(port_name=PORT, baudrate=BAUD)
    await client.set_hardware_type(SID, HW)
    await client.set_finger_unit_mode(SID, libstark.FingerUnitMode.Normalized)
    print("=" * 50)
    print("Revo2 Tune Hand (6-finger)  [ThumbAux,Thumb,Index,Middle,Ring,Pinky]")
    print("  0=ThumbAux  1=Thumb  2=Index  3=Middle  4=Ring  5=Pinky")
    print("=" * 50)
    await show(client)
    while True:
        print("\n  s=status  o=open_all  g=grip_all")
        print("  key pos  (e.g. 0 100)")
        print("  r name=record  p name=play  list/save/load  q=quit")
        print("-" * 45)
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        if cmd == "q":
            break
        elif cmd == "s":
            await show(client)
        elif cmd == "o":
            await set_list(client, [0]*6, dur=500)
            await show(client)
        elif cmd == "g":
            print("Grip all...")
            await set_list(client, [1000]*6, dur=400)
            await asyncio.sleep(0.4)
            await show(client)
        elif cmd in CTRL:
            if len(parts) < 2:
                print("Usage: key pos")
                continue
            try:
                pos = int(parts[1])
                pos = max(0, min(1000, pos))
            except:
                print("Position 0~1000")
                continue
            name, fid = CTRL[cmd]
            print(f"{name} -> {pos}")
            await set_one(client, fid, pos)
            await show(client)
        elif cmd == "r":
            name = parts[1] if len(parts) > 1 else "pose"
            status = await client.get_motor_status(SID)
            poses[name] = [status.positions[i] for i in range(6)]
            print(f"Recorded '{name}': {poses[name]}")
        elif cmd == "p":
            name = parts[1] if len(parts) > 1 else None
            if name is None:
                print("Usage: p name")
            elif name not in poses:
                print(f"No '{name}', have: {list(poses.keys())}")
            else:
                print(f"Play '{name}': {poses[name]}")
                await set_list(client, poses[name])
                await show(client)
        elif cmd == "list":
            if not poses: print("(empty)")
            for k, v in poses.items(): print(f"  {k}: {v}")
        elif cmd == "save":
            fname = parts[1] if len(parts) > 1 else "hand_poses.json"
            with open(fname, "w") as f:
                json.dump(poses, f, indent=2, ensure_ascii=False)
            print(f"Saved {fname}")
        elif cmd == "load":
            fname = parts[1] if len(parts) > 1 else "hand_poses.json"
            try:
                with open(fname, "r") as f: poses.update(json.load(f))
                print(f"Loaded {fname}")
            except FileNotFoundError:
                print(f"Not found: {fname}")
        else:
            print("?")
    libstark.modbus_close(client)
    print("Exit")

if __name__ == "__main__":
    asyncio.run(main())