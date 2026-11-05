# Sensors

Cameras are pinhole: focal length and sensor apertures in mm, image size in pixels. A camera counts only as a
`UsdGeom.Camera` prim with `arb:sensor_model` naming it and exactly these intrinsics (within 2 %); its mass counts toward the
robot's. A lidar is not a camera: declare it on any prim with `arb:sensor_model`; its mass counts, and in the bay
mission it is what lets a flying robot see the other two (range and vertical field below; 2D: only robots within
±0.25 m of its own height).

| model | type | resolution | focal [mm] | aperture h × v [mm] | range [m] | vertical field [°] | mass [kg] |
|---|---|---|---|---|---|---|---|
| `CAM-RGB-W` | rgb | 1920 × 1200 | 4.0 | 11.26 × 7.04 | - | - | 0.060 |
| `CAM-RGB-M` | rgb | 1920 × 1200 | 8.0 | 11.26 × 7.04 | - | - | 0.065 |
| `CAM-RGB-T` | rgb | 1920 × 1200 | 16.0 | 11.26 × 7.04 | - | - | 0.090 |
| `CAM-RGB-S` | rgb | 640 × 480 | 3.6 | 4.80 × 3.60 | - | - | 0.012 |
| `THERM-A` | thermal | 160 × 120 | 1.7 | 1.92 × 1.44 | - | - | 0.015 |
| `THERM-B` | thermal | 640 × 512 | 13.0 | 7.68 × 6.14 | - | - | 0.110 |
| `THERM-C` | thermal | 640 × 512 | 9.0 | 7.68 × 6.14 | - | - | 0.095 |
| `TOF-D` | depth | 320 × 240 | 2.0 | 2.88 × 2.16 | - | - | 0.040 |
| `STEREO-X` | depth | 1280 × 720 | 2.1 | 5.00 × 2.80 | - | - | 0.075 |
| `LIDAR-2D` | lidar | - | - | - | 25 | 0 | 0.160 |
| `LIDAR-3D` | lidar | - | - | - | 30 | 90 | 0.350 |
