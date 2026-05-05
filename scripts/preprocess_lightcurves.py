import pandas as pd
from pathlib import Path
import numpy as np
import sys

try:
	from astropy.io import fits
except Exception:
	fits = None

try:
	import torch
except Exception:
	torch = None

# Lee el manifiesto de descargas y filtra sólo filas con status == 'ok'
p = Path("mamba-exoplanet/data/splits/manifest.csv")
df = pd.read_csv(p)

# Normalizar el campo status por si hay espacios o mayúsculas
df["status"] = df["status"].astype(str).str.strip().str.lower()
df_ok = df[df["status"] == "ok"].copy()

print(f"Total filas: {len(df)} — filas con status 'ok': {len(df_ok)}")
print(df_ok.head())

# Guardar resultado filtrado
out = p.with_name("manifest_ok.csv")
df_ok.to_csv(out, index=False)
print(f"Manifest filtrado guardado en: {out}")

# Extraer PDCSAP_FLUX de los archivos FITS correspondientes a los TICs con status 'ok' y guardarlos como .npy
def extract_flux_quality_and_sector(fpath: Path):
	"""Extrae PDCSAP_FLUX, QUALITY (si existe) y número de sector desde el FITS.
	Retorna (flux_array, quality_array_or_None, sector_or_None, error_or_None)."""
	if fits is None:
		return None, None, None, "astropy_not_available"
	try:
		hdul = fits.open(fpath)
	except Exception as e:
		return None, None, None, f"open_error:{e}"

	try:
		sector = None
		for h in hdul:
			# intentar leer sector desde header
			hdr = getattr(h, "header", None)
			if hdr is not None and sector is None:
				for key in ("SECTOR", "SECT", "SECTNUM", "SECTORID"):
					if key in hdr:
						try:
							sector = int(hdr[key])
							break
						except Exception:
							pass
			data = h.data
			if data is None:
				continue
			try:
				cols = getattr(data, "columns", None)
				if cols is not None and "PDCSAP_FLUX" in cols.names:
					flux = np.array(data["PDCSAP_FLUX"], dtype=float)
					quality = None
					if "QUALITY" in cols.names:
						quality = np.array(data["QUALITY"], dtype=int)
					return flux, quality, sector, None
			except Exception:
				pass
			# structured numpy
			try:
				if hasattr(data, "names") and "PDCSAP_FLUX" in data.names:
					flux = np.array(data["PDCSAP_FLUX"], dtype=float)
					quality = None
					if "QUALITY" in data.names:
						quality = np.array(data["QUALITY"], dtype=int)
					return flux, quality, sector, None
			except Exception:
				pass
	finally:
		try:
			hdul.close()
		except Exception:
			pass

	return None, None, sector, "pdcsap_not_found"


def run_extraction(df_ok: pd.DataFrame):
	base_search = Path("mamba-exoplanet/data/raw/lightcurves/mastDownload/TESS")
	out_dir = Path("mamba-exoplanet/data/processed/global")
	out_dir.mkdir(parents=True, exist_ok=True)

	total_tics = len(df_ok)
	found_files = 0
	saved = 0
	errors = []

	if not base_search.exists():
		print(f"Directorio base no encontrado: {base_search}")
		return

	for tid in df_ok["tid"]:
		# buscar recursivamente archivos que contengan el TIC
		matches = list(base_search.rglob(f"*{int(tid)}*.fits"))
		if not matches:
			errors.append((tid, "no_fits_found"))
			continue

		# para cada fichero intentar extraer sector y arrays
		per_file = []  # list of (sector_or_none, flux, quality)
		for f in matches:
			found_files += 1
			flux, quality, sector, err = extract_flux_quality_and_sector(f)
			if flux is None:
				errors.append((tid, f"{f.name}:{err}"))
				continue
			per_file.append((sector if sector is not None else 1_000_000, flux, quality))

		if not per_file:
			errors.append((tid, "no_valid_pdcsap"))
			continue

		# ordenar por sector (None al final)
		per_file.sort(key=lambda x: x[0])

		# limpiar cada sector: aplicar QUALITY==0 y valores finitos
		seqs = []
		for sector_num, flux, quality in per_file:
			try:
				mask = np.isfinite(flux)
				if quality is not None:
					try:
						mask &= (np.array(quality) == 0)
					except Exception:
						pass
				clean = np.asarray(flux)[mask]
				if clean.size > 0:
					seqs.append(clean)
			except Exception as e:
				errors.append((tid, f"clean_error:{e}"))

		if not seqs:
			errors.append((tid, "all_sectors_empty_after_clean"))
			continue

		# concatenar secuencias por TIC
		try:
			full = np.concatenate(seqs)
		except Exception as e:
			errors.append((tid, f"concat_error:{e}"))
			continue

		# convertir a tensor y guardar como .pt (PyTorch)
		if torch is None:
			errors.append((tid, "torch_not_available"))
			continue
		tensor = torch.tensor(full, dtype=torch.float32)
		out_path = out_dir / f"{int(tid)}.pt"
		try:
			torch.save(tensor, out_path)
			saved += 1
			print(f"Guardado: {out_path} (len={tensor.shape[0]})")
		except Exception as e:
			errors.append((tid, f"save_error:{e}"))

	print("--- Resumen extracción PDCSAP ---")
	print(f"TICs en manifest_ok: {total_tics}")
	print(f"Archivos FITS encontrados: {found_files}")
	print(f"Arrays guardados: {saved}")
	print(f"Errores / faltantes: {len(errors)}")
	if errors:
		print(errors[:20])


if __name__ == "__main__":
	# Ejecutar extracción tras filtrar
	run_extraction(df_ok)