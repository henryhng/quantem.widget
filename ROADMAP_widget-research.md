# quantem.widget — Research-Grounded Roadmap

Synthesis of 10 parallel research agents covering py4DSTEM, HyperSpy/exspy/lumispy, napari, abTEM, AtomAI/atomap, pyxem/kikuchipy/orix, LiberTEM-live/stempy/Bluesky/Nion, scikit-image/pystackreg/cellpose/BM3D, Tomviz/3D Slicer/VTK/Dragonfly, and ImageJ/Fiji (TrackMate/BigStitcher/MorphoLibJ/omero-figure).

Every recommendation below is grounded in a specific named function / URL from an existing package. Vague "should have ML" suggestions are excluded.

---

## 1. Executive summary

`quantem.widget` is currently a strong **viewer / interaction layer** for EM data — 14 widgets with shared GPU colormap, WebGPU FFT, WebGPU volume ray-caster, GPU-decompressed Arina IO, ROI / line-profile / FFT / GIF-export protocols, and quantem-Dataset duck typing. What it lacks, relative to the established Python EM stack, is **analysis primitives** (strain, orientation, decomposition, segmentation, atom-column localization, EELS/EDS modelling), **live acquisition**, **lazy/streaming IO for > RAM data**, and a **shared layer/event model** that would let widgets cross-link cursors, ROIs, and contrast.

Three high-leverage architectural shifts unlock most of the per-widget upgrades simultaneously:

1. **Lazy / dask backend** with multiscale OME-Zarr ingest (parity with HyperSpy `LazySignal`, napari-ome-zarr, abTEM zarr, LiberTEM partitions).
2. **Shared layer + event-bus model** (napari Image/Labels/Points/Shapes/Tracks/Surface/Vectors + EventEmitter pattern, plus HyperSpy `axes_manager.events.indices_changed` cross-widget cursor sync).
3. **Crystallographic / wave / atomic-model types** as first-class duck-typed inputs alongside `Dataset` / `IOResult`: `orix.CrystalMap`, `pyxem.OrientationMap`, `abtem.Waves`, `ase.Atoms`, `dask.array.Array`.

On top of these, ~20 well-scoped new widgets fall out naturally from existing reference implementations in the ecosystem.

---

## 2. Cross-cutting themes (patterns appearing in ≥ 3 agents)

| Theme | Agents that called for it | Reference implementations |
|---|---|---|
| **Lazy / dask data backend** | HyperSpy, abTEM, LiberTEM, napari, Tomviz | `hyperspy.LazySignal.as_lazy()`, `abtem` dask graphs, `dask.array`, `napari-ome-zarr` |
| **Shared layer model** (Image/Labels/Points/Shapes/Tracks/Surface/Vectors) | napari, AtomAI, Fiji, skimage, py4DSTEM | `napari.layers.*`, ImageJ ROI Manager, atomap polarization vectors |
| **Event bus / observable model for widget→widget sync** | HyperSpy, napari, abTEM, LiberTEM | `axes_manager.events.indices_changed`, `napari.utils.events.EventEmitter` |
| **Crystallography as a typed input** (alongside numpy / Dataset) | py4DSTEM, pyxem/kikuchipy, abTEM | `orix.CrystalMap`, `pyxem.OrientationMap`, `ase.Atoms` |
| **Transfer-function editor / advanced 3D rendering** | Tomviz, napari, Dragonfly | `vtkGPUVolumeRayCastMapper`, Tomviz 1D/2D TF |
| **OME-Zarr multiscale ingest** | napari, Tomviz, abTEM, HyperSpy, Fiji | `napari-ome-zarr`, `ome-ngff`, `abtem.from_zarr` |
| **Plugin manifest pattern** | napari, Fiji, HyperSpy | `napari.yaml` (npe2), ImageJ macro, HyperSpy extensions list |
| **Streaming / partition-based processing** | LiberTEM, stempy, Bluesky | `Context.run_udf_iter`, `LiveContext`, `RE.subscribe` |
| **Cross-widget linked cursor / ROI / contrast** | HyperSpy, napari | `plot_signals(sync=True)`, evented `Layer.contrast_limits` |
| **Active learning loop on user-corrected labels** | AtomAI, napari | `EnsembleTrainer`, Empanada-napari |
| **GPU detection / fallback path unified** | abTEM, AtomAI, LiberTEM, skimage | `abtem.config["device"]`, `BasePredictor(use_gpu=...)` |

---

## 3. Per-existing-widget upgrade matrix

For each widget, the highest-value additions, grounded in real package APIs.

### Show1D
- **`find_peaks_ohaver`** Hyperspy-style 1st-derivative peak finder (`hyperspy._signals.Signal1D.find_peaks1D_ohaver(slope_thresh, amp_thresh, medfilt_radius)`). Add a `Marks` trait + sliders.
- **SpanROI integration**: `(left, right)` trait with `changed` event → integrated value. Mirrors `hyperspy.roi.SpanROI`.
- **Linked spectrum cursor** to Show2D / Show4D via the new event bus (§5.2).

### Show2D
- **Multi-channel composite** (`channel_axis`, per-channel `colormap` + `blending`) — napari pattern; allows EELS/EDS elemental maps overlaid on HAADF.
- **First-class `orix.CrystalMap` input**: detect `isinstance(data, CrystalMap)` → render via `IPFColorKeyTSL(xmap.phases[0].point_group).orientation2color(xmap.orientations)` then `reshape(xmap.shape + (3,))`.
- **`vector_field` overlay** (Nx,Ny,2) for CoM / DPC arrow maps.
- **`colormap_signed=True`** diverging cmap for ε_xx, ε_yy, CoM, etc.
- **Denoise preview toggle**: `denoise_nl_means` / `denoise_tv_chambolle` on the displayed frame only; non-destructive.
- **TrackMate-style track overlay** when fed `(track_id, t, y, x)` array.

### Show3D
- **TrackMate-style particle tracking**: per-frame detection (`skimage.feature.blob_log`) + LAP linker (`scipy.optimize.linear_sum_assignment`) + gap close + Kalman option. Reference: Tinevez 2017, `imagej.net/plugins/trackmate`.
- **Diff mode optimization** (already flagged in CLAUDE.md as a slow path) via partition-based GPU compute.
- **`axes_manager.indices` style `nav_index` + `indices_changed` event** for cross-widget cursor sync.
- **Append-frame mode** for in-situ time series streaming (LiberTEM partition adapter).

### Show3DVolume
The single biggest upgrade target. Currently has 3-orthogonal-slice ray-cast with brightness + log + colormap. Tomviz/3D Slicer/VTK provide an entire upgrade path.
- **1D transfer-function editor**: `tf_handles: List[{x: Float, opacity: Float, color: (r,g,b)}]`, sampled CPU-side into a 256-tap LUT, uploaded as a 1D texture. Replaces single brightness scalar. Ref: Tomviz "Integrated Histogram, Opacity & Color TF Editor".
- **2D transfer function** (intensity × gradient-magnitude): `tf_2d: Float32Array[H][W][4]`. Critical for separating dense / porous regions in tomograms where 1D fails. Ref: VTK MR !2809.
- **`projection_mode: Literal["composite","mip","minip","average","iso","additive"]`** — `vtkVolumeMapper::SetBlendModeTo*`.
- **`iso_threshold` + Phong shading + gradient_opacity** — `vtkVolumeProperty::ShadeOn()`.
- **`clip_planes: List[{origin, normal, enabled}]`** (≤ 6) — `vtkAbstractMapper3D::AddClippingPlane`, mind the napari z-y-x vs vispy x-y-z axis-order trap.
- **`labels_volume` + `labels_opacity`** — overlay segmentation as a second 3D texture (3D Slicer `vtkMRMLSegmentationDisplayNode`).
- **`add_volume(data_b, tf_b)`** — true multi-channel (HAADF + EELS-elemental) volume rendering.
- **`crop_box`, `slab_axis`, `slab_thickness`, `slab_op`, `attenuated_mip`** — Tomviz cropping, Slicer slab MIP, napari attenuated_mip with `attenuation` param.
- **Early-ray termination (`α > 0.99`)** and **empty-space skipping** via coarse occupancy grid.

### Show4D
- **Two `nav_index` traits + `synced_to` other-widget trait** for HyperSpy-style linked-navigator workflow.
- **Lazy / dask backend** — biggest gap (4D-STEM datasets routinely exceed RAM). Reference: HyperSpy "Working with big data".

### Show4DSTEM
The other biggest upgrade target — py4DSTEM and pyxem expose an analysis stack quantem.widget mostly lacks.
- **Bragg-peak detection + overlay**: `detect_bragg_disks(corr_power=1.0, sigma=2.0, min_rel_intensity=0.005, max_num_peaks=70, subpixel="multicorr")` wrapping `py4DSTEM.process.diskdetection.find_Bragg_disks`. Markers on the CBED canvas + `bragg_peaks` trait round-tripping `PointListArray`.
- **Origin + ellipse calibration**: `py4DSTEM.process.calibration.get_origin`/`fit_origin`/`fit_ellipse_amorphous_ring` exposed as a calibration sub-panel.
- **Polar / azimuthal sub-view**: `polar_datacube` lazy cache; on ROI drag, show `PolarDiffraction2D.get_azimuthal_integral2d(npt=256, npt_azim=180)` + 1D radial profile. Enables amorphous/medium-range-order maps live.
- **Live orientation readout**: if attached `OrientationMap` exists, hover-cursor displays `(φ₁, Φ, φ₂)` + `corr` + `plot_template_over_pattern` overlay. Ref: `pyxem.signals.OrientationMap.to_ipf_colormap`.
- **Live mode** (LiberTEM-live wrapper): subscribe `Context.run_udf_iter(dataset=aq, udf=[BFUDF, ADFUDF, HAADFUDF, SumUDF])` and push partition deltas; reuse Arina GPU pipeline as `process_partition`.
- **Drop-frame counter + dose budget** via DECTRIS frame-id deltas — safety during a 700 GB / 15 s acquisition (NCEM 4D-Camera scale).
- **HyperSpy-compatible cursor sync** to a SpectrumImage (when EELS attached as a sister modality).

### ShowComplex2D
- **`from_phase_reconstruction(recon)` constructor**: pull `recon.object_cropped` and `recon.angular_object_phase` from py4DSTEM `Parallax` / `Ptychography`.
- **`from_multislice_object(recon, component="phase")`** for `MultislicePtychography.object` (n_slices, Nx, Ny) complex.
- **Aberration tableau**: small χ(k) plot from `abtem.transfer.CTF.evaluate_chi()` driven by sliders on `Aberrations` (C10, C12, phi12, C21, …) — see AberrationExplorer below.
- **Remove-phase-ramp action** (FFT-centroid fit).

### Mark2D
- **`fit_subpixel`** action: 2D Gaussian fit per marker (atomap `Atom_Position.refine_position_using_2d_gaussian` convention: amplitude, x0, y0, sx, sy, theta, offset). Replace marker with refined position; store σ/θ as metadata for column-shape analysis.
- **`ingest_segmentation(mask | predictor)`** — accept boolean/label mask or any AtomAI `BasePredictor`-shaped object; populate markers from `regionprops(...).centroid`.
- **Active-learning hook**: emit `(patch, label)` on every marker edit; background `EnsembleTrainer` warm-starts and re-emits uncertainty map → drives a "next patch" suggester.
- **Named ROI/point IDs + groups** (ImageJ ROI Manager pattern): sortable table, color-by-group, `.zip`-roundtrippable JSON.
- **HyperSpy markers schema parity** (`Points`, `Circles`, `Lines`, `Arrows`, `Texts`, `Polygons`) so HDF5 round-trips with HyperSpy.

### Edit2D
- **napari Labels-layer semantics**: int32 label image with `paint`, `fill`, `selected_label`, `brush_size`, `preserve_labels`, `n_edit_dimensions`. Unlocks multi-class EM segmentation (vacuum / film / particle / damage).
- **Segmentation presets**: Otsu / multi-Otsu, watershed (markers from `peak_local_max` on the distance transform), SLIC + RAG merge, random-walker, Chan-Vese. All via `skimage.segmentation.*`.
- **`predict_mask(model)`** — duck-typed AtomAI predictor: `mask = model.predict(image).mask`; pre-fill brush layer.
- **`uncertainty_overlay`** from `EnsembleTrainer` variance — paints brush focus where the model is unsure.
- **`reference_mask` / `moving_mask` exposed via traits** — feeds Align2D for masked phase-cross-correlation.

### Align2D
- **`method` dropdown**: `phase_corr` (current), `pystackreg.{TRANSLATION,RIGID_BODY,SCALED_ROTATION,AFFINE,BILINEAR}`, `optical_flow_tvl1`, `optical_flow_ilk`. Drop-in via `pystackreg.StackReg(StackReg.AFFINE).register(ref, mov)`.
- **Masked phase-cross-correlation**: `skimage.registration.phase_cross_correlation(ref, mov, reference_mask=..., moving_mask=...)` — handles detector gaps / saturated regions.
- **Flow vector overlay** when `optical_flow_tvl1` selected.
- **`from_parallax(recon)`** constructor reusing `Parallax._xy_shifts` displacement field.

### Bin2D
- **Calibrated axes** (HyperSpy `axes_manager`): bin factor reflects scaled units.
- **Polar binning**: when fed a `PolarDatacube`, bin in (q, θ) directly.

### Bin4D
- **Sparse-array fast path** for electron-counted data (`stempy.io.sparse_array.SparseArray`) — bin without densifying.

### MetricExplorer
- **`metrics` dict trait** accepting `{name: (Nx, Ny)}` arrays from py4DSTEM `BraggVectors.peaks.length`, ACOM `correlation_max`, polar radial-peak position, etc. Already PCA-adjacent — thin client on `decomposition().learning_results`.
- **Live mode (Bluesky-style)**: each scan point is an `event` doc; subscribe a `(name, doc)` callback → append to a `LivePlot` trace. Surface `start/descriptor/event/stop` hooks as widget events.

### Browse
- **Bio-Formats-style multi-file series detection**: pattern-detect `*_001.dm3`, `*_002.dm3`, … → preview a series picker with per-frame metadata (angle, exposure) before loading. Tilt-series, focal series, multi-position scans.
- **py4DSTEM calibration tab**: origin, q-pixel-size, ellipse a/b/θ. Mutates `DataCube.calibration` in place.
- **OMERO / iRODS read adapters** on top of `omero.gateway.BlitzGateway` and `python-irodsclient`.
- **`compute_navigator()` thumbnail** when fed a `LazySignal` — no full materialization.

---

## 4. Proposed new widgets — prioritized

### Tier 1 — high microscopy value, builds on existing infra

**1. `AtomFinder`** — STEM HAADF/iDPC atom-column localizer with sub-pixel Gaussian fit + sublattice partitioning + polarization arrows. Traits: `atom_positions: (N,2)`, `sublattice_a/b: indices`, `polarization_vectors: (M,4)`, `fit_gaussian_subpixel`. Reference: `atomap.make_atom_lattice_from_image` + `Sublattice.refine_atom_positions_using_2d_gaussian` + `get_polarization_from_second_sublattice`. Microscopy value: live ferroelectric polarization mapping in BaTiO₃/PbTiO₃, domain-wall screening in real time.

**2. `StrainMap2D`** — 4-channel viewer for ε_xx, ε_yy, ε_xy, θ from a reference-region picker. Calls `BraggVectors.choose_lattice_vectors` → `fit_lattice_vectors_masked` → `calculate_strain`. Diverging cmap, shared colorbar, 2D vector overlay of (g₁, g₂). Reference: py4DSTEM `strain_mapping_*.ipynb`. Value: in-situ heating, mechanical-loading, interface strain — watch ε evolve frame-by-frame on 5D data.

**3. `SpectrumImage`** — EELS/EDS map+spectrum dual viewer. Show2D of integrated map + Show1D of spectrum at cursor, linked by shared cursor event. Traits: `integration_window=(e0, e1)`, `map_mode={"sum","max","argmax"}`. Backed by `exspy.signals.EELSSpectrum` / `EDSTEMSpectrum`. Value: routine EELS/EDS workflow, today done in matplotlib nbagg with painful responsiveness.

**4. `AberrationExplorer`** — live abTEM probe / CTF diagnostics. Traits: `energy_keV`, `semiangle_cutoff_mrad`, `aberrations: dict[str, float]` (C10..C34, phi12..phi34), `defocus_spread_A`. Panels: probe intensity (`Probe.build().intensity()`), aberration phase wheel from `abtem.transfer.CTF.evaluate_chi()`, 1D radial CTF, `TemporalEnvelope` damping. Value: real-time Ronchigram-style aberration tuning at the column.

**5. `TransferFunctionEditor`** — 1D and 2D TF editor for Show3DVolume. 2D mode plots (intensity, |∇I|) histogram with paintable opacity field → Float32Array RGBA texture. Reference: Tomviz, VTK MR !2809. Value: separates dense / porous regions in reconstructed tomograms where a 1D slider cannot.

### Tier 2 — deeper analysis modules

**6. `ACOMOrientationMap`** — IPF-colored orientation map from `py4DSTEM.Crystal.match_orientations`. Brushable correlation threshold, click-to-show diffraction with `plot_template_over_pattern` overlay, grain-cluster toggle. Traits: `crystal` (Crystal pickle), `orientation_map`, `corr_threshold`, `ipf_axis`. Reference: Ophus et al. arXiv:2111.00171. Value: identify grain boundaries and twin variants live in polycrystalline scans.

**7. `Decomposition`** — interactive PCA / NMF / ICA / ORPCA explorer. Traits: `algorithm`, `n_components`, `centre`, `normalize_poissonian_noise`. Shows scree (`plot_explained_variance_ratio`), loadings grid, factor stack; brush-select components → `signal.get_decomposition_model(components=[...])`. Reference: HyperSpy MVA user guide. Value: blind chemical-component separation for spectrum images — the single most-requested HyperSpy GUI feature historically.

**8. `ParallaxReconstructor`** — wraps `py4DSTEM.process.phase.Parallax.preprocess() → .reconstruct() → .aberration_fit() → .aberration_correct()` as a stepped UI. Delegates shift-display to Align2D. Outputs aberrations table (C1, A1, C3). Value: post-acquisition defocus/aberration estimate + dose-efficient BF image without a full iterative ptycho.

**9. `TrackMate`** — particle/spot tracker over Show3D-style stacks. Pipeline: DoG/LoG/`peak_local_max` detect → Jaqaman LAP link (`scipy.optimize.linear_sum_assignment`) → gap close → optional Kalman predictor. Renders `Tracks` overlay (napari schema) with tail length + color-by-track-id. Emits per-track time series to Show1D for intensity / displacement / MSD plots. Ref: Tinevez 2017, `imagej.net/plugins/trackmate`.

**10. `Segment2D` / `Segment3D`** — Labels-layer widget with brush + classical-presets + DL-presets:
- Classical: Otsu, multi-Otsu, watershed, SLIC+RAG-merge, random-walker, Chan-Vese (all `skimage.segmentation.*`).
- DL (gated by try-import): Cellpose-SAM (`cellpose.models.CellposeModel(...).eval()`), AtomAI `Segmentor`.
Outputs an instance-label volume; paintable + mergeable. Reference: napari Labels API + scikit-image segmentation user guide.

**11. `Denoise2D` / `Denoise3D`** — split-view raw vs denoised with method dropdown (Gaussian, Wiener, NLM, BM3D, TV-Chambolle, Bilateral, Wavelet), per-method sliders, auto-fill `sigma = skimage.restoration.estimate_sigma`, live PSNR/SSIM. BM3D via `pip install bm3d` (try-import). Value: pre-acquisition parameter tuning — pick on one frame, apply to a series.

### Tier 3 — acquisition / pipeline / 3D

**12. `LiveAcquire`** — LiberTEM-live front-door. Detector dropdown: `merlin | dectris | asi_tpx3 | memory`. Default passive mode: `wait_for_acquisition` → `make_acquisition(pending_aq=...)`. Runs BF/ADF/HAADF UDFs + `PartitionMonitorUDF`; pipes partition results into a Show4DSTEM child. Drop-frame counter from DECTRIS monotone frame-id gaps. Ref: `libertem_live.api.LiveContext`, SIMPLON 1.8 ZMQ port 9999 PUSH/PULL.

**13. `DriftCorrect`** — N-image rigid/affine/non-rigid stack alignment. `pystackreg.StackReg(...).register_transform_stack(stack, reference='previous'|'first'|'mean', moving_average=k)`. Per-frame diagnostic plot (tx, ty, θ over frame), outlier flagging (`|shift| > k·MAD`). Two-way bind frame-slider to Show3D for scrubbing aligned + raw side-by-side.

**14. `TomographyAlign`** — tilt-series xcorr alignment + manual fiducial drag + tilt-axis handles (fiducial circular ⇔ aligned, crescent ⇔ misaligned, per Tomviz manual). Outputs aligned stack ready for ASTRA Toolbox (`astra.creators.create_reconstruction`, `SIRT3D_CUDA`, `CGLS3D_CUDA`) → feeds Show3DVolume.

**15. `Stitcher`** — phase-correlation 2D mosaic builder for STEM scan tiles. Pairwise translations from inverse FFT of normalized cross-power spectrum (Kuglin-Hines 1975), confidence filter, global least-squares optimization, linear blending fuse. Overlap preview overlay before fusion. Ref: Preibisch 2009 / BigStitcher.

**16. `AtomLattice`** — post-localization plane / strain analyzer; consumes `atom_positions` from `AtomFinder`. Traits: `zone_vector_idx`, `selected_plane_id`, `line_scan_profile`, `monolayer_distance_map`, `angle_map`, `strain_gradient_map`. Reference: `atomap.construct_zone_axes_from_sublattice`, `get_monolayer_distance_map`, `get_atom_distance_difference_map`. Value: per-plane line-scans for local strain at heterointerfaces, dislocation cores, ferroelastic twin boundaries.

**17. `DLSegmentation`** — drop-in U-Net inference panel; loads a local AtomAI state-dict, overlays predicted mask on Show2D in real time. `aoi.models.load_model(weights_path)` / `load_ensemble`. Traits: `weights_path`, `nb_classes`, `device`, `threshold`, `refine`, `mask`, `coords`, `uncertainty`. Coords flow into Mark2D for human verification — closes the active-learning loop.

**18. `PolarDiffraction`** — radial × azimuth viewer with mask-able rings (drag radial band). `pyxem.signals.PolarDiffraction2D.get_azimuthal_integral2d`; per-ROI azimuth profile, ring intensity vs scan position, RDF export (`py4DSTEM.process.rdf.get_radial_distribution_function`). Optional Ewald-curvature correction. Value: amorphous / metallic-glass / polymer-blend medium-range-order maps live at the microscope.

### Tier 4 — infrastructure / quality of life

**19. `LazyShow4D` / lazy backend across widgets** — Dask-backed Show4D for > RAM 4D-STEM (typical Merlin/EMPAD = 50–500 GB). Traits: `chunks`, `navigator` (pre-computed thumbnail via `compute_navigator()`), `prefetch_chunks`. On `nav_index_changed`, only touched chunk materializes. Ref: HyperSpy `LazySignal`, `as_lazy()`, "Working with big data".

**20. `ModelFit`** — interactive HyperSpy `Model1D` editor. Traits: `components=[{type:"PowerLaw", params:{A, r}}, ...]`, `fit_range`, `bounded`. Visual residual panel below the fit. Hooks `model.fit()`, `model.multifit()`. Reference: `hyperspy.model.BaseModel`, `hs.model.components1D`, `exspy.components.EELSCLEdge`.

**21. `ExpVsSim`** — exp 4D-STEM ↔ abTEM-simulated 4D-STEM with NCC goodness-of-fit map per scan pixel. Parameter sweep (defocus, thickness, tilt) tied to MetricExplorer. abTEM `SMatrix(potential, energy, planewave_cutoff, interpolation).scan(grid, detectors=PixelatedDetector())` — PRISM is the right engine because the same S-matrix is reused across the sweep.

**22. `IsoSurfaceViewer`** — `vtkFlyingEdges3D`-equivalent marching-cubes; live threshold, multi-contour list, per-contour color + opacity, mesh export (PLY/STL/glTF) for downstream metrology.

**23. `VolumeMeasure`** — pickable 3D point/line/region on Show3DVolume. Length in calibrated Å, region mean/std intensity, segmented region volume. Analog of 3D Slicer Markups + Segment Statistics.

**24. `FigureBuilder`** — omero-figure-style multi-page panel builder. Each open widget contributes "snapshot" panels; drag onto grid canvas, edit scalebar/label/panel-letter, export multi-page PDF at 300 dpi. Ref: `omero-figure` `Figure_To_Pdf.py`.

**25. `MacroRecorder`** — subscribe to every widget's state-change events, emit Python that re-creates them (`w = Mark2D(...); w.add_point(...); w.set_view(...)`). Output modes: Python / JSON. Ref: ImageJ Recorder.

**26. `BookkeepingExplorer`** — Bluesky-style `RunStart/Descriptor/Event/Stop` log viewer + scrubber timeline + "what changed in this window" diff. Synthesize doc shapes from microscope telemetry (focus, stage, beam, vacuum) and LiveContext hook callbacks. Ref: nsls-ii.github.io/event-model.

**27. `SafetyMonitor`** — subscribes to live partition stream; cheap UDFs compute (a) BF mean drift, (b) CoM drift as rigid-body proxy, (c) dose budget, (d) saturation pixel fraction. Tripwire fires JS toast + Notification API + `on_abort` trait. Target: abort a bad 700 GB / 15 s acquisition before it costs an hour.

**28. `PluginShelf`** — manifest-driven plugin host. Static `quantem.yaml` mirroring npe2 `contributions: {commands, widgets, readers, writers, sample_data, menus}`. Discovery via `importlib.metadata` entry points. Encourages an ecosystem like Empanada/TARDIS-em/brainglobe.

**29. `AxesManagerHub` / `LinkedNavigators`** — invisible plumbing widget holding `(indices, axes_calibration)` and broadcasting `indices_changed`. Lets a `SpectrumImage`'s Show1D, Show2D, and `Decomposition` viewer share one cursor. The HyperSpy `plot_signals(sync=True)` analog.

**30. `EBSDIndexer`** — kikuchipy dictionary indexing UI. Hough mode (`EBSD.hough_indexing(phase_list, indexer)` wraps PyEBSDIndex), Dictionary mode (`master_pattern.get_patterns(...)` then `EBSD.dictionary_indexing(dict, metric="ncc")`, optional `refine_orientation`). Outputs `CrystalMap` consumed by upgraded Show2D.

---

## 5. General architecture improvements

These are the foundations that unlock most of §3 and §4 simultaneously.

### 5.1 Lazy / dask backend

Accept `dask.array.Array` anywhere `np.ndarray` is currently demanded. Render navigator from `LazySignal.compute_navigator()` rather than full `.compute()`. Honor the no-in-place-ops invariant. Adds first-class support for:
- `> RAM 4D-STEM` (HyperSpy big-data guide)
- abTEM Dask graphs of simulations
- LiberTEM partition iteration
- `napari-ome-zarr` multiscale pyramids
- Future LazyShow4D, LazyShow3DVolume

### 5.2 Shared layer model + event bus

Promote ROI / markers / profiles / labels to a top-level `LayerList` typed by `{Image, Labels, Points, Shapes, Tracks, Surface, Vectors}` (napari `napari.components.ViewerModel`). Any widget renders any layer it understands. Each layer exposes evented attributes (`layer.events.<name>.connect(callback)` per napari `EmitterGroup`). The HyperSpy `axes_manager.events.indices_changed` is the same primitive applied to cursor indices. With this in place:
- Cross-widget linked cursor (Show2D ↔ Show4D ↔ SpectrumImage)
- Linked contrast across multi-channel composites
- Mark2D's `Points` showing on Show3D, on Show4DSTEM scan-space, in a Show1D track plot, simultaneously
- napari → quantem.widget annotation interop (export `Points` JSON, load in napari)

### 5.3 Crystallographic / wave / atomic-model duck typing

Extend the existing `IOResult` / `Dataset` duck-typed input protocol to recognize:
- `orix.crystal_map.CrystalMap` → auto-render via `IPFColorKeyTSL.orientation2color(.orientations)`, expose `xmap.prop` (scores, CI, fit) as auto-discovered scalar layers
- `pyxem.signals.OrientationMap` / `Diffraction2D` / `PolarDiffraction2D` → auto-detect polar axes (`signal.axes_manager.signal_axes[0].name == "Radius"`) → polar sub-view
- `abtem.waves.Waves` → ShowComplex2D auto-route
- `ase.Atoms` → expose to AberrationExplorer / `Potential` for ExpVsSim
- `py4DSTEM.BraggVectors`, `StrainMap`, `Probe`, `Crystal` → typed routes

### 5.4 Plugin manifest pattern

Static `quantem.yaml` listing `commands`, `widgets`, `readers`, `writers`, `sample_data`, `themes`, `menus`. Discovery via `importlib.metadata` entry points (avoid the npe1 → npe2 migration pain napari hit). Same shape as npe2 `WidgetContribution`. Reference: napari.org/plugins/contributions.html.

### 5.5 OME-Zarr / multiscale ingest

OME-NGFF as a first-class IO route. `napari-ome-zarr`-equivalent reader picks LOD by screen-space voxel size. Required once tomograms exceed ~1024³ (cryo-ET whole-cell ≈ 4–8 k³). Also lands abTEM's `from_zarr` and HyperSpy big-data path on the same backend.

### 5.6 Async streaming channel

Wrap `Context.run_udf_iter` in a Python `async for` and ship **partition deltas** over the anywidget message channel (msgpack + zero-copy buffers). Frontend acks each partition; Python coroutine awaits ack with a budget → coalesces under back-pressure (same pattern LiberTEM's `PipelinedExecutor` uses internally). Underpins LiveAcquire, append-frame Show3D, Bluesky-style MetricExplorer.

### 5.7 Unified GPU / device probe

Single arbiter for CuPy (abTEM, py4DSTEM), torch (AtomAI, Arina pipeline), WebGPU (ColormapEngine, FFT). Expose as `device: "auto" | "cuda" | "mps" | "cpu" | "wgpu"` per-widget trait. `BasePredictor(use_gpu=...)`, `abtem.config.set({"device": "gpu"})`, `getWebGPUFFT()` all flow through the same probe.

### 5.8 Session-level annotations store

Single JSON envelope (points, ROIs, tracks, labels, units, calibration) every widget subscribes to. ImageJ ROI Manager + napari LayerList generalized. Survives a kernel restart; the bridge between Mark2D, Edit2D, Show3D-tracks, FigureBuilder.

### 5.9 Animation timeline

napari-animation `KeyFrame(viewer_state, steps, ease)` model + ffmpeg export — fly-throughs, HRSTEM time-series figures, presentation-quality videos straight from a notebook.

### 5.10 Active-learning loop primitive

Three pieces wired together: (Edit2D | Mark2D edits) → emit `(patch, label)` event → background `EnsembleTrainer` warm-starts on a queue → emit fresh prediction + per-pixel uncertainty → drive a "next patch to label" suggester. AtomAI's `dklGPR.thompson` and ensemble variance are the references.

---

## 6. Prioritization rubric

Sorted by (experiment-risk reduction × build-feasibility), highest first:

1. **§5.1 Lazy/dask backend** — unblocks all > RAM workflows; risk to live experiments today is "cannot open the data we just acquired".
2. **§5.2 Shared layer + event bus** — enables most multi-widget workflows that microscopists describe verbally as one task.
3. **AtomFinder + AtomLattice + DLSegmentation** — directly enable polarization mapping and defect screening during a session.
4. **StrainMap2D + ACOMOrientationMap + ParallaxReconstructor** — the py4DSTEM analysis stack that quantem.widget currently lacks.
5. **SpectrumImage + Decomposition + ModelFit + AxesManagerHub** — the HyperSpy EELS/EDS workflow.
6. **TransferFunctionEditor + IsoSurfaceViewer + Show3DVolume upgrades** — unlocks tomography QC.
7. **LiveAcquire + SafetyMonitor + Async streaming channel** — bring acquisition-time QC in.
8. **DriftCorrect + Segment + Denoise** — the scikit-image / pystackreg utility belt.
9. **TrackMate + Stitcher + TomographyAlign + ASTRA hook** — the Fiji parity.
10. **PluginShelf + FigureBuilder + MacroRecorder + BookkeepingExplorer** — ecosystem & reproducibility infrastructure.

A microscopist running a polycrystalline-in-situ experiment today benefits from items 1–4 well before items 9–10.

---

## 7. Caveats and non-goals

- **WebGPU testing limit**: per CLAUDE.md, neither Playwright Chromium nor system Chrome exposes `navigator.gpu` on most CI machines. Any new GPU path (transfer function editor, multi-volume blending, ray-cast Phong) still needs Safari/JupyterLab human verification.
- **No-hard-dependency contract**: every Tier-1 widget can take a numpy/torch/cupy/Dataset input via duck typing; DL backends (cellpose, AtomAI Segmentor, BM3D) gate behind `try-import` so the base install stays light.
- **Behavior preservation**: where we wrap an existing package, the wrapper exposes parameters with the same names and defaults as the reference — `pystackreg.StackReg` constants, `napari.layers.Labels` knobs, `abtem.transfer.Aberrations` Krivanek symbols. Microscopists already know these.
- **Out of scope**: vendor-specific commercial integrations (Velox SDK, DigitalMicrograph plugin packaging, Avizo file format) — only the open hooks (Gatan DM Python `GMSLive2DPlot`, Bio-Formats `.dm3`) are entertained.

---

## 8. Sources

py4DSTEM
- https://github.com/py4dstem/py4DSTEM
- https://py4dstem.readthedocs.io/en/latest/apiindex.html
- https://ar5iv.labs.arxiv.org/html/2111.00171
- https://academic.oup.com/mam/article/27/4/712/6888063
- https://github.com/py4dstem/py4DSTEM_tutorials

HyperSpy / exspy / lumispy
- https://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html
- https://hyperspy.org/hyperspy-doc/current/user_guide/axes.html
- https://hyperspy.org/hyperspy-doc/current/reference/base_classes/roi.html
- https://hyperspy.org/hyperspy-doc/v1.7/user_guide/mva.html
- https://hyperspy.org/hyperspy-doc/current/user_guide/big_data.html
- https://exspy.readthedocs.io/en/v0.2.1/user_guide/eels.html

napari
- https://napari.org/stable/guides/layers.html
- https://napari.org/api/napari.layers.Labels.html
- https://napari.org/dev/api/napari.utils.events.html
- https://napari.org/plugins/contributions.html
- https://napari.org/0.4.16/gallery/volume_plane_rendering.html
- https://github.com/napari/napari-animation
- https://github.com/ome/napari-ome-zarr

abTEM
- https://abtem.readthedocs.io/en/latest/getting_started/overview.html
- https://abtem.readthedocs.io/en/latest/reference/api/_autosummary/abtem.waves.Probe.html
- https://abtem.readthedocs.io/en/latest/reference/api/_autosummary/abtem.transfer.CTF.html
- https://abtem.readthedocs.io/en/latest/user_guide/tutorials/prism.html
- https://github.com/abTEM/abTEM

AtomAI / atomap
- https://github.com/pycroscopy/atomai
- https://atomai.readthedocs.io/en/latest/atomai_models.html
- https://atomap.org/api_documentation.html
- https://www.nature.com/articles/s41598-021-84499-w (AtomSegNet)
- https://www.nature.com/articles/s41524-024-01360-0

pyxem / kikuchipy / orix
- https://www.pyxem.org/en/v0.21.0/examples/processing/azimuthal_integration.html
- https://www.pyxem.org/en/latest/reference/generated/pyxem.signals.OrientationMap.html
- https://github.com/pyxem/pyxem/blob/main/pyxem/utils/indexation_utils.py
- https://kikuchipy.org/en/stable/tutorials/pattern_matching.html
- https://orix.readthedocs.io/en/stable/reference/generated/orix.crystal_map.CrystalMap.plot.html
- https://orix.readthedocs.io/en/stable/tutorials/inverse_pole_figures.html

LiberTEM / stempy / DECTRIS / Nion / Bluesky
- https://libertem.github.io/LiberTEM/udf/introduction.html
- https://libertem.github.io/LiberTEM-live/reference.html
- https://libertem.github.io/LiberTEM-live/detectors/merlin.html
- https://libertem.github.io/LiberTEM-live/reference/dectris.html
- https://media.dectris.com/filer_public/6d/57/6d5779b4-2c8c-45a7-8792-6ef447f1ddde/simplon_apireference_v1p8.pdf
- https://github.com/OpenChemistry/stempy
- https://arxiv.org/html/2407.03215v1
- https://nionswift.readthedocs.io/en/stable/api/hardware.html
- https://nsls-ii.github.io/bluesky/callbacks.html
- https://nsls-ii.github.io/event-model/data-model.html

scikit-image / pystackreg / cellpose / BM3D
- https://scikit-image.org/docs/stable/api/skimage.registration.html
- https://scikit-image.org/docs/stable/api/skimage.restoration.html
- https://scikit-image.org/docs/stable/api/skimage.segmentation.html
- https://pystackreg.readthedocs.io/en/latest/readme.html
- https://cellpose.readthedocs.io/en/latest/
- https://pypi.org/project/bm3d/

Tomviz / 3D Slicer / VTK / Dragonfly / napari 3D
- https://tomviz.readthedocs.io/en/latest/visualization/
- https://tomviz.readthedocs.io/en/latest/alignment/
- https://www.kitware.com/2d-transfer-function-support-in-gpuvolumemapper/
- https://gitlab.kitware.com/vtk/vtk/-/merge_requests/2809
- https://vtk.org/doc/nightly/html/classvtkGPUVolumeRayCastMapper.html
- https://slicer.readthedocs.io/en/latest/user_guide/modules/volumerendering.html
- https://napari.org/guides/rendering-explanation.html
- https://www.sciencedirect.com/science/article/abs/pii/S0304399115001060 (ASTRA)

ImageJ / Fiji / TrackMate / BigStitcher / omero-figure
- https://imagej.net/plugins/trackmate/
- https://github.com/trackmate-sc/TrackMate
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2747604/ (Jaqaman LAP, Nature Methods 2008)
- https://imagej.net/plugins/bigstitcher/
- https://publications.mpi-cbg.de/Preibisch_2009_1199.pdf (BigStitcher / phase correlation)
- https://github.com/ome/omero-figure
- https://imagej.net/plugins/morpholibj
- https://academic.oup.com/bioinformatics/article/32/22/3532/2525592 (MorphoLibJ)
- https://github.com/astra-toolbox/astra-toolbox

---

## What's next

1. **Prototype `AtomFinder` (Tier 1)** on a real HAADF dataset — atom-column localization + sub-pixel Gaussian fit + sublattice partition. Reuses Mark2D's point primitive + atomap math. *Experiment value:* live polarization mapping in ferroelectrics during a session; the single most-requested capability microscopists ask for that quantem.widget cannot do today.
2. **Stand up the lazy / dask backend (§5.1)** with a `LazyShow4D` MVP — accept `dask.array.Array`, render the navigator via `compute_navigator()`, materialize only the touched chunk on `nav_index_changed`. *Experiment value:* removes the "cannot open the file we just acquired" failure mode for ≥ 50 GB 4D-STEM, which today silently aborts a session.
3. **Wire the event bus (§5.2) end-to-end on a single linked pair** — Show2D ↔ SpectrumImage with cross-cursor sync via a HyperSpy-style `indices_changed` emitter. Once this works for two widgets, every Tier-2 widget inherits cursor/contrast linking. *Experiment value:* EELS-at-cursor on a STEM image, today done in matplotlib nbagg with seconds of lag.
4. **Add `bragg_disk_detection` + `polar_datacube` to Show4DSTEM** (py4DSTEM hooks). Lowest-friction path to Tier-2 strain and orientation analysis. *Experiment value:* identifies bad / drifting calibration during acquisition before a multi-hour scan is committed to.
5. **Land the `TransferFunctionEditor` for Show3DVolume** — 1D editor first, 2D when needed. *Experiment value:* tomography QC operators currently bounce to Tomviz/Dragonfly mid-session because quantem cannot separate dense vs porous regions in a reconstruction.

## Files modified

- `ROADMAP_widget-research.md` (new, ~610 lines, this file)
