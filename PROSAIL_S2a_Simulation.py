# Notwendige Bibliotheken werden immer zuerst importiert. vgl. "https://pypi.org/project/prosail/"
import pandas as pd
import numpy as np
from prosail import run_prosail


# Der Dateipfad, wo die finale CSV-Datei gespeichert werden soll.
output_filename = r'C:\Users\benny\Desktop\BR41\Google Earth Engine\Zeitreihe\LAI\PROSAIL_S2A_TRAINING\PROSAIL_Simulationen_S2_Training.csv'
# Anzahl der Durchläufe. Jeder Durchlauf erzeugt eine Zeile in unserer finalen Tabelle.
num_simulations = 90000

# Das Wellenlängen-Array wird hier erstellt, da es für jede Simulation gleich ist.
wavelengths = np.arange(400, 2501)
# Eine leere Liste, in der wir die Ergebnisse jedes einzelnen Durchlaufs sammeln.
results_list = []

# Definition der exakten Spezifikationen der Sentinel-2A Bänder.
# Format: (Zentrale Wellenlänge in nm, Halbwertsbreite FWHM in nm)
s2a_bands_for_resampling = np.array([
    (442.7, 21), (492.4, 66), (559.8, 36), (664.6, 31),
    (704.1, 15), (740.5, 15), (782.8, 20), (864.7, 21),
    (945.1, 20), (1613.7, 91), (2202.4, 175)
])
# Liste der Spaltennamen für unsere finale CSV-Datei.
s2a_band_column_names = [
    'S2A_B1', 'S2A_B2', 'S2A_B3', 'S2A_B4', 'S2A_B5', 'S2A_B6', 'S2A_B7',
    'S2A_B8', 'S2A_B8A', 'S2A_B9', 'S2A_B11', 'S2A_B12'
]

# Berechnet die Reflexion für Sentinel-2-Bänder aus hochaufgelösten PROSAIL-Daten.
def resample_reflectance(prosail_wl, prosail_rho, s2_bands):
    s2_reflectance = []
    for center_wl, fwhm in s2_bands:
        srf = np.exp(-0.5 * ((prosail_wl - center_wl) / (fwhm / 2.355)) ** 2)
        band_reflectance = np.sum(prosail_rho * srf) / np.sum(srf)
        s2_reflectance.append(band_reflectance)
    return np.array(s2_reflectance)

print(f"Starte {num_simulations} PROSAIL-Simulationen")
for i in range(num_simulations):
    # Parameter für diesen einen Durchlauf zufällig festlegen Parameter Angaben: vgl. "https://doi.org/10.1080/17538947.2025.2496403"
    n = np.random.uniform(1, 3)                       # N: Blattstruktur-Parameter
    cm = (10 * n - 9) / (1000 * n + 250)              # Cm: Trockenmasse pro Fläche (g/cm2)
    cab = np.random.uniform(10, 80)                   # cab: Chlorophyll a+b Gehalt (ug/cm2)
    car = np.random.uniform(0, 30)                    # car: Karotinoid-Gehalt (ug/cm2)
    cbrown = np.random.uniform(0, 1)                  # cbrown: Gehalt an braunen Pigmenten
    cw = np.random.uniform(0.02)                      # Cw: Äquivalente Wasserdicke (cm)
    psoil = np.random.uniform(0, 1)                   # psoil: Trocken/Nass-Bodenfaktor
    rsoil = 1                                         # rsoil: Bodenhelligkeitsfaktor
    lai = np.random.uniform(0.1, 8.0)                 # LAI: Blattflächenindex
    lidfa = np.random.uniform(0, 80)                  # lidfa: Durchschnittlicher Blattneigungswinkel (Grad)
    hspot = np.random.uniform(0.01)                   # hspot: Hotspot-Parameter
    # tts_min = 27.66
    # tts_max = 74.38
    tts = np.random.uniform(27.66, 74.38)
    tto = 0.0                                         # tto: Beobachter-Zenitwinkel (Grad)
    psi = 0.0                                         # psi: Relativer Azimutwinkel (Grad)

    # PROSAIL-Modell ausführen
    reflectance_raw = run_prosail(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto, psi, psoil=psoil,
                                  rsoil=rsoil)

    # Auf Sentinel-2 Bänder umrechnen, indem wir unsere oben definierte Funktion AUFRUFEN
    s2_reflectance = resample_reflectance(wavelengths, reflectance_raw, s2a_bands_for_resampling)

    # B8A-Wert einfügen
    s2_reflectance_12_bands = np.insert(s2_reflectance, 8, s2_reflectance[7])

    # Ergebnisse für diesen Durchlauf speichern
    result_row = {
        'lai': lai, 'cab': cab, 'car': car, 'cbrown': cbrown,
        'cw': cw, 'cm': cm, 'n': n, 'lidfa': lidfa,
        'hspot': hspot, 'tts': tts, 'psoil': psoil
    }
    for band_name, reflectance_value in zip(s2a_band_column_names, s2_reflectance_12_bands):
        result_row[band_name] = reflectance_value

    results_list.append(result_row)

    # Fortschrittsanzeige
    if (i + 1) % 1000 == 0:
        print(f"Simulation {i + 1}/{num_simulations} abgeschlossen.")

print("Alle Simulationen abgeschlossen.")

# Ergebnisse in eine CSV-Datei speichern
df = pd.DataFrame(results_list)
df.to_csv(output_filename, index=False)

print(f"Trainingsdaten erfolgreich in '{output_filename}' gespeichert.")
