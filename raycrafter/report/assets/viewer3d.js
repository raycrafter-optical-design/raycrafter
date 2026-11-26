// The 3D view of the RayCrafter HTML report (see raycrafter/report/viewer_3d.py). It draws the
// scene embedded in the page, in the format described in raycrafter/core/scene.py, with
// three.js. It looks and fits like the app's 3D view (raycrafter/gui/view_3d.py).
import * as THREE from 'three';
import { OrbitControls } from 'three-orbit-controls';

const FIELD_OF_VIEW_DEG = 30;   // horizontal
const FIT_MARGIN = 1.08;        // room left round the scene when a view is chosen
const MAX_ELEVATION_DEG = 89.9; // straight down would leave "which way is up?" undecided

const box = document.getElementById('viewer3d');
const message = document.getElementById('viewer3d-message');
const data = JSON.parse(document.getElementById('scene-data').textContent);

// The axis runs across the screen: the bench's (x, y, z) are drawn at (z, x, y), z being up.
function toView(flat) {
  const points = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 3) {
    points.set([flat[i + 2], flat[i], flat[i + 1]], i);
  }
  return points;
}

function geometryOf(flat) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(toView(flat), 3));
  return geometry;
}

// One object of the scene: its translucent meshes, which ignore depth so that glass never hides
// what is behind it, and its solid polylines.
function drawingsOf(item) {
  const drawings = [];
  const surface = new THREE.MeshBasicMaterial({
    color: item.colour, opacity: item.opacity, transparent: true, side: THREE.DoubleSide,
    depthTest: false, depthWrite: false,
  });
  for (const mesh of item.meshes) {
    const geometry = geometryOf(mesh.vertices);
    geometry.setIndex(mesh.faces);
    drawings.push(new THREE.Mesh(geometry, surface));
  }
  const stroke = new THREE.LineBasicMaterial({ color: item.colour });
  for (const line of item.lines) {
    drawings.push(new THREE.Line(geometryOf(line), stroke));
  }
  return drawings;
}

function show() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor('#ffffff');
  box.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  for (const item of data.objects) scene.add(...drawingsOf(item));
  const bounds = new THREE.Box3().setFromObject(scene);
  const centre = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());

  const camera = new THREE.PerspectiveCamera();
  camera.up.set(0, 0, 1);
  camera.near = size.length() / 1000;
  camera.far = size.length() * 100;
  const controls = new OrbitControls(camera, renderer.domElement);
  const render = () => renderer.render(scene, camera);
  controls.addEventListener('change', render);

  // Look from one of the standard views, as far away as it takes to see the whole bounding box.
  function setView(name) {
    const [azimuth, elevation] = data.views[name].map(THREE.MathUtils.degToRad);
    const limit = THREE.MathUtils.degToRad(MAX_ELEVATION_DEG);
    const height = Math.min(elevation, limit);
    const toCamera = new THREE.Vector3(Math.cos(height) * Math.cos(azimuth),
      Math.cos(height) * Math.sin(azimuth), Math.sin(height));
    const right = new THREE.Vector3(-Math.sin(azimuth), Math.cos(azimuth), 0);
    const up = new THREE.Vector3().crossVectors(toCamera, right);
    const tanHalfWidth = Math.tan(THREE.MathUtils.degToRad(FIELD_OF_VIEW_DEG) / 2);
    const tanHalfHeight = tanHalfWidth / camera.aspect;
    let distance = 0;
    for (let corner = 0; corner < 8; corner++) {
      const offset = new THREE.Vector3((corner & 1 ? 0.5 : -0.5) * size.x,
        (corner & 2 ? 0.5 : -0.5) * size.y, (corner & 4 ? 0.5 : -0.5) * size.z);
      distance = Math.max(distance, offset.dot(toCamera) + Math.max(
        Math.abs(offset.dot(right)) / tanHalfWidth, Math.abs(offset.dot(up)) / tanHalfHeight));
    }
    camera.position.copy(centre).addScaledVector(toCamera, FIT_MARGIN * distance);
    controls.target.copy(centre);
    controls.update();
    render();
  }

  function resize() {
    const width = box.clientWidth, height = box.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    // three.js takes the vertical angle; the horizontal one is what stays fixed here.
    const tanHalfWidth = Math.tan(THREE.MathUtils.degToRad(FIELD_OF_VIEW_DEG) / 2);
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(tanHalfWidth / camera.aspect));
    camera.updateProjectionMatrix();
    render();
  }

  resize();
  setView('3D');
  new ResizeObserver(resize).observe(box);
  document.querySelectorAll('[data-view]').forEach(button => {
    button.addEventListener('click', () => setView(button.dataset.view));
  });
  message.hidden = true;
}

try {
  show();
} catch (error) {
  message.textContent = 'The 3D view needs a browser with WebGL, which is not available here.';
}
