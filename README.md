# s2-ltae-unet

Pixel-wise forecasting of post-disturbance spectral change from Sentinel-2 L2A image time series with a temporal-attention U-Net.

## Methods

- STAC search of Sentinel-2 L2A cloud-optimised GeoTIFFs; per-scene BOA offset; SCL and haze (HOT) masking; monthly median NBR, NDVI and NDMI
- dNBR burn severity, FIREMON classes; MODIS burned-area and active-fire burn dates
- deseasonalised NBR; per-pixel exponential recovery model fitted by grid search; rate constant k, T80, Relative Recovery Index (RRI); year-on-year breakpoint detection
- terrain covariates (GLO-30 DEM, D8 flow accumulation, TWI), tree cover, SPEI-12
- L-TAE + U-Net, plain U-Net, ConvLSTM (PyTorch); LightGBM, persistence and exponential baselines
- spatial block cross-validation, paired block bootstrap

## Usage

Copy `config.example.json` to `config.json`, fill in the values, or set `RS_CONFIG=/path/to/config.json`.

```
python s2_download.py
python s2_reduce.py --data s2_data
python compute_dnbr.py s2_data
python compute_fire_progression.py s2_data
python compute_recovery.py s2_data out/recovery
python compute_controls.py out/recovery/recovery_100m.tif out/controls
python compute_controls_stats.py out/controls out/recovery out/controls
python model_prep.py s2_data/reduced out/recovery/recovery_100m.tif s2_data/fire_pixels.csv out/inputs.npz
python train_cv.py out/inputs.npz {tae|plain|convlstm} FOLD out/model BUDGET_S [CHECKPOINT]
python baselines_cv.py out/inputs.npz out/model
python cv_summary.py out/inputs.npz out/model out/model
python fig_unet_architecture.py out/fig
python fig_model_performance.py out/inputs.npz out/model out/fig
python fig_topographic_controls.py out/controls out/fig
```

Python 3.14, PyTorch 2.14.1, LightGBM 4.7.0.

## License

MIT
