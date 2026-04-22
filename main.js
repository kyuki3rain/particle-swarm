// ---- WebGPU Particle Swarm Simulation ----
// Compute shader updates positions/velocities; render shader draws particles as points.

const NUM_PARTICLES = 32768; // 2^15
const WORKGROUP_SIZE = 256;

// ---- UI wiring ----
const sliders = {
  local:   { el: document.getElementById('sl-local'),   val: document.getElementById('val-local'),   v: 0.5 },
  individ: { el: document.getElementById('sl-individ'), val: document.getElementById('val-individ'), v: 0.5 },
  sync:    { el: document.getElementById('sl-sync'),    val: document.getElementById('val-sync'),    v: 0.5 },
  global:  { el: document.getElementById('sl-global'),  val: document.getElementById('val-global'),  v: 0.5 },
};
for (const [k, s] of Object.entries(sliders)) {
  s.el.addEventListener('input', () => {
    s.v = parseFloat(s.el.value);
    s.val.textContent = s.v.toFixed(2);
  });
}

// Panel toggle
const panelBody = document.getElementById('panel-body');
const toggleBtn = document.getElementById('toggle-btn');
document.getElementById('panel-header').addEventListener('click', () => {
  const open = panelBody.style.display !== 'none';
  panelBody.style.display = open ? 'none' : '';
  toggleBtn.textContent = open ? '▼' : '▲';
});

// ---- Presets ----
// Lerp runs in its own rAF loop, fully independent of WebGPU init timing.
const PRESETS = {
  'small-animal': { local: 0.8, individ: 0.8, sync: 0.1, global: 0.0 },
  'swarm':        { local: 0.3, individ: 0.2, sync: 0.9, global: 0.1 },
  'grand':        { local: 0.1, individ: 0.1, sync: 0.9, global: 0.9 },
  'curious':      { local: 0.7, individ: 0.5, sync: 0.2, global: 0.1 },
  'default':      { local: 0.5, individ: 0.5, sync: 0.5, global: 0.5 },
};
const PRESET_KEYS = ['local', 'individ', 'sync', 'global'];
const LERP_MS = 550;

let lerpHandle = null;
let activePresetBtn = null;

function applyPreset(preset) {
  if (lerpHandle !== null) cancelAnimationFrame(lerpHandle);
  const from = {};
  for (const k of PRESET_KEYS) from[k] = sliders[k].v;
  const startMs = performance.now();

  (function step() {
    const t = Math.min(1.0, (performance.now() - startMs) / LERP_MS);
    const ease = t * t * (3 - 2 * t); // smoothstep
    for (const k of PRESET_KEYS) {
      const v = from[k] + (preset[k] - from[k]) * ease;
      sliders[k].v = v;
      sliders[k].el.value = v.toFixed(2);
      sliders[k].val.textContent = v.toFixed(2);
    }
    lerpHandle = t < 1.0 ? requestAnimationFrame(step) : null;
  })();
}

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const preset = PRESETS[btn.dataset.preset];
    if (!preset) return;
    applyPreset(preset);
    if (activePresetBtn) activePresetBtn.classList.remove('active');
    activePresetBtn = btn;
    btn.classList.add('active');
  });
});

// Deactivate preset when user manually moves a slider
for (const s of Object.values(sliders)) {
  s.el.addEventListener('input', () => {
    if (lerpHandle !== null) { cancelAnimationFrame(lerpHandle); lerpHandle = null; }
    if (activePresetBtn) { activePresetBtn.classList.remove('active'); activePresetBtn = null; }
  });
}

// Mouse / touch tracking
let mouseX = 0.5, mouseY = 0.5, mouseDown = false;
const canvas = document.getElementById('canvas');
canvas.addEventListener('mousemove', e => {
  mouseX = e.clientX / window.innerWidth;
  mouseY = e.clientY / window.innerHeight;
});
canvas.addEventListener('mousedown', () => mouseDown = true);
canvas.addEventListener('mouseup',   () => mouseDown = false);
canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  mouseX = e.touches[0].clientX / window.innerWidth;
  mouseY = e.touches[0].clientY / window.innerHeight;
}, { passive: false });

// ---- WebGPU init ----
async function init() {
  if (!navigator.gpu) {
    document.getElementById('webgpu-warn').style.display = 'block';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    document.getElementById('webgpu-warn').style.display = 'block';
    return;
  }
  const device = await adapter.requestDevice();

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  function resizeCanvas() {
    canvas.width  = window.innerWidth  * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;
    ctx.configure({ device, format, alphaMode: 'premultiplied' });
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // ---- Particle buffer layout (per particle, 8 floats = 32 bytes) ----
  // pos.xy, vel.xy, phase, seed, life, _pad
  const PARTICLE_STRIDE = 8; // floats
  const particleData = new Float32Array(NUM_PARTICLES * PARTICLE_STRIDE);
  for (let i = 0; i < NUM_PARTICLES; i++) {
    const b = i * PARTICLE_STRIDE;
    particleData[b + 0] = Math.random(); // pos.x
    particleData[b + 1] = Math.random(); // pos.y
    particleData[b + 2] = (Math.random() - 0.5) * 0.002; // vel.x
    particleData[b + 3] = (Math.random() - 0.5) * 0.002; // vel.y
    particleData[b + 4] = Math.random() * Math.PI * 2;    // phase
    particleData[b + 5] = Math.random();                   // seed
    particleData[b + 6] = Math.random();                   // life
    particleData[b + 7] = 0;
  }

  const particleBuf = device.createBuffer({
    size: particleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuf, 0, particleData);

  // ---- Uniform buffer (48 bytes) ----
  // mouse.xy, time, dt, localReact, individ, sync, globalScale, screenSize.xy, _pad.xy
  const uniformBuf = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---- Compute shader (WGSL) ----
  const computeShaderCode = /* wgsl */`
struct Particle {
  pos   : vec2<f32>,
  vel   : vec2<f32>,
  phase : f32,
  seed  : f32,
  life  : f32,
  pad   : f32,
}

struct Uniforms {
  mouse      : vec2<f32>,
  time       : f32,
  dt         : f32,
  localReact : f32,
  individ    : f32,
  sync       : f32,
  globalScale: f32,
  screenSize : vec2<f32>,
  mouseDown  : f32,
  _pad       : f32,
}

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> u : Uniforms;

fn hash(v: f32) -> f32 {
  return fract(sin(v * 127.1 + 311.7) * 43758.5453);
}
fn hash2(v: vec2<f32>) -> f32 {
  return fract(sin(dot(v, vec2<f32>(127.1, 311.7))) * 43758.5453);
}
fn noise2(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    sin(p.x * 3.7 + p.y * 2.1 + u.time * 0.7),
    cos(p.x * 2.3 - p.y * 3.9 + u.time * 0.5)
  );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${NUM_PARTICLES}u) { return; }

  var p = particles[idx];
  let dt = u.dt;
  let t  = u.time;

  // --- cursor force ---
  // No hard cutoff: pure inverse-square decay (gravity style).
  // Near particles feel it strongly; far ones barely.
  // Per-particle sensitivity modulated by individ:
  //   individ=0 → all react equally (sensitivity=1)
  //   individ=1 → uniform [0, 2] spread (some ignore, some react double)
  let toMouse = u.mouse - p.pos;
  let dist = length(toMouse) + 0.0001;
  let dir = toMouse / dist;
  let sensitivity = mix(1.0, hash(p.seed * 7.3) * 2.0, u.individ);
  let forceSign = select(-1.0, 1.0, u.mouseDown > 0.5); // repel default, attract on hold
  let cursorForce = dir * forceSign * sensitivity * u.localReact * 0.000018 / (dist * dist + 0.002);

  // --- trajectory jitter (個体差) ---
  let jitter = noise2(p.pos * 10.0 + p.seed * 100.0) * u.individ * 0.0004;

  // --- sync vector field (same-direction pull) ---
  let fieldAngle = sin(p.pos.x * 4.0 + t * 0.3) * cos(p.pos.y * 4.0 - t * 0.2) * 3.14159;
  let fieldVec = vec2<f32>(cos(fieldAngle), sin(fieldAngle));
  let velLen = length(p.vel);
  let velNorm = select(vec2<f32>(0.0), p.vel / velLen, velLen > 0.0001);
  let syncForce = (fieldVec - velNorm) * u.sync * 0.0003;

  // --- global pulsation (大域脈動) ---
  let pulse = sin(t * 1.5 + p.phase) * u.globalScale;
  let center = vec2<f32>(0.5, 0.5);
  let toCenter = center - p.pos;
  let pulseForce = toCenter * pulse * 0.0002;

  // --- phase advance ---
  p.phase += dt * (1.0 + u.sync * 2.0 + hash(p.seed + t) * u.individ * 0.5);

  // --- integrate ---
  let accel = cursorForce + jitter + syncForce + pulseForce;
  p.vel = p.vel * (1.0 - 0.015) + accel;

  // speed clamp
  let spd = length(p.vel);
  let maxSpd = 0.008 * (1.0 + u.globalScale * 0.5);
  if (spd > maxSpd) {
    p.vel = p.vel * (maxSpd / spd);
  }

  p.pos = p.pos + p.vel;

  // toroidal wrap
  p.pos = p.pos % vec2<f32>(1.0);
  if (p.pos.x < 0.0) { p.pos.x += 1.0; }
  if (p.pos.y < 0.0) { p.pos.y += 1.0; }

  particles[idx] = p;
}
`;

  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: computeShaderCode }),
      entryPoint: 'main',
    },
  });

  // ---- Render shader (WGSL) ----
  // Each particle rendered as a 2-triangle quad (6 vertices).
  // vi / 6 = particle index, vi % 6 = corner index within quad.
  const renderShaderCode = /* wgsl */`
struct Particle {
  pos   : vec2<f32>,
  vel   : vec2<f32>,
  phase : f32,
  seed  : f32,
  life  : f32,
  pad   : f32,
}
struct Uniforms {
  mouse      : vec2<f32>,
  time       : f32,
  dt         : f32,
  localReact : f32,
  individ    : f32,
  sync       : f32,
  globalScale: f32,
  screenSize : vec2<f32>,  // CSS pixels (innerWidth, innerHeight)
  mouseDown  : f32,
  _pad       : f32,
}

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> u : Uniforms;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) color     : vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  let pidx = vi / 6u;
  let ci   = vi % 6u;
  // Inline quad corners (triangle-list, 2 triangles CCW):
  // tri0: (0,0)(-1,-1) (1,0)(1,-1) (2,0)(-1,1)
  // tri1: (3,0)(1,-1)  (4,0)(1,1)  (5,0)(-1,1)
  var qx = -1.0; if (ci == 1u || ci == 3u || ci == 4u) { qx = 1.0; }
  var qy = -1.0; if (ci == 2u || ci == 4u || ci == 5u) { qy = 1.0; }
  let corner = vec2<f32>(qx, qy);
  let p = particles[pidx];

  // center in NDC (Y flipped: screen Y=0 is top)
  let cx = p.pos.x * 2.0 - 1.0;
  let cy = -(p.pos.y * 2.0 - 1.0);

  // 3 CSS-pixel radius quad
  let r = 3.0;
  let sx = r * 2.0 / u.screenSize.x;
  let sy = r * 2.0 / u.screenSize.y;

  var out : VSOut;
  out.pos = vec4<f32>(cx + corner.x * sx, cy + corner.y * sy, 0.0, 1.0);

  // color: hue from phase + speed tint
  let spd = length(p.vel) * 300.0;
  let hue = fract(p.phase * 0.159 + u.globalScale * 0.3);
  let sat = 0.6 + u.sync * 0.4;
  let val = 0.6 + spd * 0.4;

  let h6 = hue * 6.0;
  let i  = floor(h6);
  let f  = h6 - i;
  let q  = val * (1.0 - sat * f);
  let t2 = val * (1.0 - sat * (1.0 - f));
  let p2 = val * (1.0 - sat);
  var rgb : vec3<f32>;
  switch (i32(i) % 6) {
    case 0: { rgb = vec3<f32>(val, t2,  p2);  }
    case 1: { rgb = vec3<f32>(q,   val, p2);  }
    case 2: { rgb = vec3<f32>(p2,  val, t2);  }
    case 3: { rgb = vec3<f32>(p2,  q,   val); }
    case 4: { rgb = vec3<f32>(t2,  p2,  val); }
    default:{ rgb = vec3<f32>(val, p2,  q);   }
  }
  out.color = vec4<f32>(rgb, 0.75);
  return out;
}

@fragment
fn fs_main(@location(0) color : vec4<f32>) -> @location(0) vec4<f32> {
  return color;
}
`;

  const renderModule = device.createShaderModule({ code: renderShaderCode });
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex:   { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{
      format,
      blend: {
        color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
      }
    }]},
    primitive: { topology: 'triangle-list' },
  });

  // ---- Bind groups ----
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuf } },
      { binding: 1, resource: { buffer: uniformBuf  } },
    ],
  });
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuf } },
      { binding: 1, resource: { buffer: uniformBuf  } },
    ],
  });

  // ---- Render loop ----
  const fpsEl = document.getElementById('fps');
  let lastTime = performance.now();
  let frameCount = 0, fpsTimer = 0;

  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    frameCount++;
    fpsTimer += dt;
    if (fpsTimer >= 0.5) {
      fpsEl.textContent = `FPS: ${Math.round(frameCount / fpsTimer)}`;
      frameCount = 0; fpsTimer = 0;
    }

    // update uniforms
    const uni = new Float32Array(12);
    uni[0] = mouseX;
    uni[1] = mouseY;
    uni[2] = now / 1000;
    uni[3] = dt;
    uni[4] = sliders.local.v;
    uni[5] = sliders.individ.v;
    uni[6] = sliders.sync.v;
    uni[7] = sliders.global.v;
    uni[8] = window.innerWidth;
    uni[9] = window.innerHeight;
    uni[10] = mouseDown ? 1.0 : 0.0;
    uni[11] = 0;
    device.queue.writeBuffer(uniformBuf, 0, uni);

    const cmd = device.createCommandEncoder();

    // compute pass
    const cp = cmd.beginComputePass();
    cp.setPipeline(computePipeline);
    cp.setBindGroup(0, computeBindGroup);
    cp.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / WORKGROUP_SIZE));
    cp.end();

    // render pass
    const texture = ctx.getCurrentTexture();
    const rp = cmd.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp:  'clear',
        clearValue: { r: 0.0, g: 0.0, b: 0.04, a: 1.0 },
        storeOp: 'store',
      }],
    });
    rp.setPipeline(renderPipeline);
    rp.setBindGroup(0, renderBindGroup);
    rp.draw(NUM_PARTICLES * 6);
    rp.end();

    device.queue.submit([cmd.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

init().catch(err => {
  console.error(err);
  document.getElementById('webgpu-warn').style.display = 'block';
  document.getElementById('webgpu-warn').innerHTML += `<br><small>${err}</small>`;
});
