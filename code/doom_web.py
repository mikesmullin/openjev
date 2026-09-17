#!/usr/bin/env python
"""doom_live.py rendered into a browser page instead of a native ViZDoom window.

The game runs headless; each tic the screen buffer and the current telemetry are published to a small
stdlib HTTP server. The page shows the same things the recorded mp4 HUD showed -- frame, per-hypothesis
P(entailment), chosen action, premise text, rolling probability chart -- except live.

Two endpoints, no dependencies beyond PIL:
  /stream.mjpg   multipart/x-mixed-replace, consumed by a plain <img>
  /events        server-sent events, one JSON telemetry record per decision

    python doom_web.py --scenario defend_the_center --profile aim
    python doom_web.py --scenario freedoom2 --map map01
"""
import argparse
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import vizdoom as vzd
from PIL import Image

import doom_live as DL

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>openjev plays Doom</title><style>
*{box-sizing:border-box} body{margin:0;background:#0b1519;color:#e4efec;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid #24404a}
h1{margin:0;font-size:17px} .sub{color:#8aa6ab;font-size:12px;margin-top:2px}
main{display:flex;gap:18px;padding:18px;align-items:flex-start;flex-wrap:wrap}
#screen{background:#000;border:1px solid #24404a;border-radius:6px;display:block;width:800px;max-width:100%}
aside{flex:1;min-width:380px;display:flex;flex-direction:column;gap:14px}
.card{background:#122229;border:1px solid #24404a;border-radius:6px;padding:12px 14px}
.k{color:#8aa6ab;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.stat b{display:block;font-size:20px;font-weight:600} .stat span{color:#8aa6ab;font-size:11px}
.row{display:flex;align-items:center;gap:8px;margin:5px 0}
.bar{flex:1;height:14px;background:#0b1519;border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;width:0;transition:width .08s linear}
.val{width:42px;text-align:right;color:#8aa6ab;font-size:12px}
.hyp{flex:0 0 58%;font-size:12px;color:#c9dcd9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.act{font-size:11px;color:#8aa6ab}
#chosen{font-size:17px;font-weight:600;margin-top:10px}
#premise{color:#8aa6ab;font-size:12.5px;white-space:pre-wrap}
#chart{width:100%;height:150px;display:block}
.dead{color:#e5533d}
#ctl{margin-top:10px;display:flex;gap:8px}
#ctl button{font:inherit;font-size:12px;color:#e4efec;background:#122229;border:1px solid #24404a;border-radius:4px;padding:5px 12px;cursor:pointer}
#ctl button:hover{background:#1b3038;border-color:#3fbf7f}
</style></head><body>
<header><h1>openjev &mdash; Qwen3.5-4B NLI cross-encoder plays Doom</h1>
<div class="sub" id="sub">connecting&hellip;</div>
<div id="ctl"><button onclick="cmd('reset')">restart episode</button><button onclick="cmd('skip')">next episode</button><button id="pz" onclick="cmd('pause')">pause</button></div></header>
<main>
<img id="screen" src="/stream.mjpg" alt="Doom">
<aside>
  <div class="card"><div class="k">state</div><div class="stats">
    <div class="stat"><b id="kills">0</b><span>kills</span></div>
    <div class="stat"><b id="health">0</b><span>health</span></div>
    <div class="stat"><b id="ammo">0</b><span>ammo</span></div>
    <div class="stat"><b id="ep">0</b><span>episode</span></div>
  </div></div>
  <div class="card"><div class="k">P(entailment) per hypothesis</div>
    <div id="bars"></div><div id="chosen">&mdash;</div></div>
  <div class="card"><div class="k">last 150 decisions</div><canvas id="chart"></canvas></div>
  <div class="card"><div class="k">timing</div><div class="stats">
    <div class="stat"><b id="lat">0</b><span>ms / decision</span></div>
    <div class="stat"><b id="dps">0</b><span>decisions / s</span></div>
    <div class="stat"><b id="tps">0</b><span>tics / s</span></div>
    <div class="stat"><b id="budget">0</b><span>% of 114ms budget</span></div>
  </div></div>
  <div class="card"><div class="k">premise sent to the model</div><div id="premise"></div></div>
</aside></main>
<script>
const COLS=["#f2b134","#3fbf7f","#e5533d","#4aa3df","#c678dd","#56b6c2","#d19a66","#98c379","#e06c75"];
let hyps=null, hist=[], nh=0;
const $=id=>document.getElementById(id);
function build(h){
  hyps=h; nh=h.length; $("bars").innerHTML="";
  h.forEach((x,i)=>{
    const d=document.createElement("div"); d.className="row";
    d.innerHTML=`<span class="hyp" title="${x.text}">${x.text}</span>
      <span class="bar"><i id="b${i}" style="background:${COLS[i%COLS.length]}"></i></span>
      <span class="val" id="v${i}">0.00</span>`;
    $("bars").appendChild(d);
    const a=document.createElement("div"); a.className="act";
    a.style.cssText="margin:-4px 0 6px 6px"; a.textContent="→ "+x.action;
    a.style.color=COLS[i%COLS.length]; $("bars").appendChild(a);
  });
}
function draw(){
  const c=$("chart"), d=c.getContext("2d"), W=c.width=c.clientWidth*2, H=c.height=300;
  d.scale(1,1); d.clearRect(0,0,W,H);
  d.strokeStyle="#24404a"; d.lineWidth=2;
  [0,.5,1].forEach(y=>{d.beginPath();d.moveTo(0,H-y*H);d.lineTo(W,H-y*H);d.stroke();});
  if(hist.length<2) return;
  const n=150, s=Math.max(0,hist.length-n);
  for(let j=0;j<nh;j++){
    d.beginPath(); d.strokeStyle=COLS[j%COLS.length]; d.lineWidth=3;
    for(let i=s;i<hist.length;i++){
      const x=(i-s)/n*W, y=H-hist[i][j]*H;
      i===s?d.moveTo(x,y):d.lineTo(x,y);
    }
    d.stroke();
  }
}
async function cmd(c){ const r=await fetch("/cmd?do="+c); const j=await r.json();
  document.getElementById("pz").textContent=j.paused?"resume":"pause";
  if(c!=="pause") hist=[]; }
document.addEventListener("keydown",e=>{ if(e.key==="r")cmd("reset"); if(e.key==="n")cmd("skip"); if(e.key===" "){e.preventDefault();cmd("pause");} });
const es=new EventSource("/events");
es.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(!hyps) { build(m.hyps); $("sub").textContent=m.title; }
  $("kills").textContent=m.kills; $("ammo").textContent=m.ammo; $("ep").textContent=m.episode;
  const hp=$("health"); hp.textContent=m.health; hp.className=m.health<35?"dead":"";
  m.probs.forEach((p,i)=>{ $("b"+i).style.width=(p*100).toFixed(1)+"%"; $("v"+i).textContent=p.toFixed(2); });
  $("chosen").textContent="chosen: "+m.action.toUpperCase();
  $("chosen").style.color=COLS[m.choice%COLS.length];
  $("lat").textContent=m.lat.toFixed(0); $("dps").textContent=m.dps.toFixed(1);
  $("tps").textContent=m.tps.toFixed(1); $("budget").textContent=(m.lat/114*100).toFixed(0);
  $("premise").textContent=m.premise;
  hist.push(m.probs); if(hist.length>400) hist=hist.slice(-200);
  draw();
};
es.onerror=()=>{ $("sub").textContent="disconnected — the run has ended"; };
</script></body></html>"""


class Control:
    """Commands from the page. The game loop polls these; HTTP handler threads only set them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset = False      # restart the current episode, same seed
        self.skip = False       # abandon it and move to the next seed
        self.paused = False
        self.quit = False

    def set(self, cmd):
        with self.lock:
            if cmd == "reset":
                self.reset = True
            elif cmd == "skip":
                self.skip = True
            elif cmd == "pause":
                self.paused = not self.paused
            elif cmd == "quit":
                self.quit = True
            return {"paused": self.paused}

    def take(self):
        """Consume any pending episode-ending command."""
        with self.lock:
            r, s, self.reset, self.skip = self.reset, self.skip, False, False
            return r, s, self.paused, self.quit


class Live:
    """Latest frame + telemetry, published by the game loop and read by HTTP handler threads."""

    def __init__(self):
        self.cv = threading.Condition()
        self.jpeg = None
        self.telemetry = None
        self.gen = 0
        self.done = False

    def publish(self, frame, telemetry):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=72)
        with self.cv:
            self.jpeg, self.telemetry, self.gen = buf.getvalue(), telemetry, self.gen + 1
            self.cv.notify_all()

    def finish(self):
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def wait(self, seen):
        with self.cv:
            while self.gen == seen and not self.done:
                self.cv.wait(1.0)
            return self.jpeg, self.telemetry, self.gen, self.done


class Handler(BaseHTTPRequestHandler):
    live = None
    ctrl = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            return self.stream()
        if self.path.startswith("/events"):
            return self.events()
        if self.path.startswith("/cmd"):
            cmd = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("do", [""])[0]
            body = json.dumps(self.ctrl.set(cmd)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                jpeg, _, seen, done = self.live.wait(seen)
                if done:
                    return
                if jpeg is None:
                    continue
                self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                _, tel, seen, done = self.live.wait(seen)
                if done:
                    return
                if tel is not None:
                    self.wfile.write(b"data: " + json.dumps(tel).encode() + b"\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../openjev_hf/qwen3.5-4b-nli")
    ap.add_argument("--scenario", default="defend_the_center", choices=list(DL.SCENARIOS))
    ap.add_argument("--profile", default=None, choices=list(DL.PROFILES))
    ap.add_argument("--map", default=None)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticrate", type=int, default=35)
    ap.add_argument("--timeout", type=int, default=2100)
    ap.add_argument("--res", default="RES_640X480")
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--port", type=int, default=8732)
    args = ap.parse_args()
    args.no_window = True

    scen = DL.SCENARIOS[args.scenario]
    profile = args.profile or scen["profile"]
    game = DL.make_game(scen, args, profile, window=False)
    hyps, actions, vectors = DL.build_actions(game, profile)
    title = args.scenario.replace("_", " ") + (f" {args.map or scen.get('map','')}" if "iwad" in scen else "")
    world = DL.World(game, title, sorted(set(actions)))

    print(f"scenario {args.scenario}  profile {profile}  ({len(hyps)} hypotheses)")
    brain = DL.Brain(DL.Policy(args.ckpt, hyps), len(hyps))
    for _ in range(3):
        brain.policy("Doom. No enemies are visible right now.")
    brain.start()

    live = Live()
    ctrl = Control()
    Handler.live = live
    Handler.ctrl = ctrl
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n    open  http://127.0.0.1:{args.port}/\n", flush=True)

    meta = [{"text": h, "action": a} for h, a in zip(hyps, actions)]
    kills, ep, interrupted = [], 0, False
    try:
        while ep < args.episodes:
            game.set_seed(args.seed + 100 + ep)
            game.new_episode()
            world.trail = []
            t0, n0, tics = time.perf_counter(), brain.latest()[3], 0
            reset = skip = quit_ = False
            while not game.is_episode_finished():
                reset, skip, paused, quit_ = ctrl.take()
                if reset or skip or quit_:
                    break
                if paused:
                    time.sleep(0.05)     # hold the world still; ASYNC only advances when we ask it to
                    continue
                s = world.observe()
                if s is None:
                    break
                premise = world.render(s)
                brain.submit(premise)
                i, p, lat, n = brain.latest()
                game.set_action(vectors[i])
                game.advance_action(1)
                tics += 1
                el = time.perf_counter() - t0
                live.publish(game.get_state().screen_buffer if game.get_state() else np.zeros((10, 10, 3), np.uint8), {
                    "title": f"{title} · profile {profile} · {len(hyps)} hypotheses scored per decision",
                    "hyps": meta, "probs": [float(x) for x in p], "choice": i, "action": actions[i],
                    "kills": s["kills"], "health": s["health"], "ammo": s["ammo"], "episode": ep,
                    "lat": lat, "dps": (n - n0) / max(el, 1e-9), "tps": tics / max(el, 1e-9),
                    "premise": premise,
                })
            if quit_:
                break
            if reset:
                print(f"episode {ep}: restarted from the page", flush=True)
                continue             # same seed, so ep is deliberately not incremented
            k = int(world.var("KILLCOUNT"))
            if not skip:
                kills.append(k)
            print(f"episode {ep}: {k} kills  {tics} tics in {time.perf_counter()-t0:.1f}s "
                  f"({tics/max(time.perf_counter()-t0,1e-9):.1f} tics/s){' [skipped]' if skip else ''}", flush=True)
            ep += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        brain.stop_flag = True
        brain.wake.set()
        brain.join(timeout=5)
        live.finish()
        game.close()
    if kills:
        print(f"kills mean {np.mean(kills):.2f} max {max(kills)} over {len(kills)} episodes")


if __name__ == "__main__":
    main()
