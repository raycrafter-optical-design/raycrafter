# three.js, as used by the report's 3D view

These files are from [three.js](https://threejs.org) **0.160.0** (MIT licence, see `LICENSE`),
copied unchanged from the npm package, so that the HTML report works without an internet
connection:

| File here             | File in the package                          |
| --------------------- | -------------------------------------------- |
| `three.module.min.js` | `build/three.module.min.js`                  |
| `OrbitControls.js`    | `examples/jsm/controls/OrbitControls.js`     |
| `LICENSE`             | `LICENSE`                                    |

Do not edit them. To update, take the same files from a newer package
(<https://registry.npmjs.org/three>, checking its `dist.integrity` hash). From version 0.172 the
library is split into more than one file, which `raycrafter/report/viewer_3d.py` would have to
allow for.
