# micro_vlbi

These Python scripts were used in the analysis of data presented in the Radio Science paper "First observations with a GNSS antenna to radio telescope interferometer."

The large FITS data files corresponding to the three experiments are available [here](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/VVN3VP).

### Scripts

#### `sensitivity_analyzer.py`
This script takes input flux density, observed SNR, and elevation to compute the observed system temperature for a dish-GNSS antenna interferometer.

**Example usage:**
```bash
python sensitivity_analyzer.py --bandwidth 39.6e6 --start_freq 1380.8e6 --input_name ./data/source_data_uy001_nvss.txt --output_name uy001_sensitivity.txt --antenna_file ./data/antenna_pattern.csv
```

#### `sim_vis_nvss.py`
Simulate radioastronomical 'visibilities' for a specified observation scenario using the NVSS catalog. Generate several figures showing the components of the source on the sky and the flux density in different scenarios.

**Usage:**
```bash
python sim_vis_nvss.py --rxname1 FD_VLBA --rxname2 DBR205 --rxpos1 "-1324009.3502 -5332181.9482 3231962.3802" --rxpos2 "-1324095.2788 -5332177.7083 3231908.2627" --time "2022-01-26T13:21:55" --catalog ./data/NVSS_cat_pared.txt --centerFreqHz 1440e6 --bandwidth 72e6 --genfringe --scanLen 600 --ra 212.8359583 --dec 52.2025 --Npt 200 --source "3C295" --searchRad 10
```

#### `clock_correct_ppp.py`
Correct the phase of complex visibilities in a FITS file with data from a PPP clock solution. Note that FITS files are not included in the GitHub repository but can be obtained from the TDL link above.

**Usage:**
```bash
python clock_correct_ppp.py --ant1 FD_VLBA --ant2 DBR205 --pppclockfile uy001b_ppp_data_corrdbr205.txt --filename ./data/uy001b1.fits --fileout ./data/uy001b1_corrected.fits
```


#### `find_nsource.py`
Find the number of visible sources in the NVSS catalog for given interferometer characteristics. The system temperature of the paired radio telescope is assumed to be that of FD-VLBA.

**Usage:**
```bash
python python find_nsource.py --antenna_file ./data/antenna_gain.csv --catalog_file ./data/NVSS_cat_pared.txt --bandwidth 200e6 --start_freq 1376e6 --rad 10 --SNR_lim 10 --t_obs 5 --Tsys_antenna 200
```
