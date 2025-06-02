print('LAI SKRIPT STARTPUNKT - Random Forest Ansatz');

// ----------------------------------------------------------------------------------
// Benutzerdefinierte Konfigurationen
// ----------------------------------------------------------------------------------

// Pfad zur CSV-Datei mit den Simulationsdaten (Trainingsdaten)
var simulationAssetPath = 'projects/red-splice-451512-s6/assets/PROSAIL_Simulationen_S2_Training';
// Pfad zum Untersuchungsgebiet
var einzugsgebietAssetPath = 'projects/red-splice-451512-s6/assets/Einzugsgebiet_Earth_Engine_BR41';

// Sentinel-2 Collection ID
var s2_collection_id = 'COPERNICUS/S2_SR_HARMONIZED'; // Harmonized Surface Reflectance
// Alternativ für Level-2A Produkte: 'COPERNICUS/S2_L2A'

// Spaltennamen in der CSV-Simulationsdatei für die Zielvariable und wichtige Eingangsmerkmale
var laiPropertyInSimulations = 'lai'; // Spaltenname für LAI-Werte
var szaPropertyInSimulations = 'tts'; // Spaltenname für Solar Zenith Angle (SZA)

// Mapping von internen Merkmalsnamen zu Spaltennamen der Reflektanzbänder in der CSV
// Dies stellt sicher, dass die korrekten Spalten aus der CSV für die entsprechenden Merkmale gelesen werden.
var reflectanceColumnsInSimulations = {
  'blue': 'S2A_B2',   // Internes Merkmal 'blue' wird aus CSV-Spalte 'S2A_B2' gelesen
  'green': 'S2A_B3',  // Internes Merkmal 'green' wird aus CSV-Spalte 'S2A_B3' gelesen
  'red': 'S2A_B4',    // Internes Merkmal 'red' wird aus CSV-Spalte 'S2A_B4' gelesen
  'NIR': 'S2A_B8A'  // Internes Merkmal 'NIR' wird aus CSV-Spalte 'S2A_B8A' gelesen (B8A empfohlen für Vegetation)
};

// Standardisierte Merkmalsnamen, die im Modelltraining und bei der Bildvorhersage verwendet werden
var inputFeatureNames = ['SZA', 'blue', 'green', 'red', 'NIR'];

// Ursprüngliche Sentinel-2 Bandnamen, die für die Merkmale verwendet werden
var s2OriginalBands = ['B2', 'B3', 'B4', 'B8A']; // Sentinel-2 Bänder für Blau, Grün, Rot, Nahinfrarot (B8A)
// Ziel-Bandnamen nach Umbenennung in prepareImageForPrediction (müssen mit reflectanceColumnsInSimulations-Keys und inputFeatureNames übereinstimmen)
var s2FeatureBands = ['blue', 'green', 'red', 'NIR'];

// Skalierungsfaktor für Sentinel-2 Reflektanzwerte (üblicherweise 0.0001 für SR Produkte)
var bandScaling = 0.0001;

// Parameter für das Random Forest Modell
var numberOfTrees = 50; // Anzahl der Bäume im Random Forest
var randomSeed = 0;     // Optional: Seed für reproduzierbare Trainingsergebnisse

// Einstellungen für die monatliche Verarbeitung und den Export
var startDate = ee.Date('2017-04-01');
var endDate = ee.Date('2025-01-01'); // Verarbeitet Monate bis VOR diesem Datum (d.h. bis Dezember 2024)
var outputDriveFolder = 'BR41_LAI_2017_2024'; // Name des Zielordners in Google Drive
var exportScale = 10; // Exportauflösung in Metern (passend zu S2 10m Bändern)
var exportCRS = 'EPSG:32632'; // Ziel-Koordinatensystem für den Export

// ----------------------------------------------------------------------------------
// Ende Benutzerdefinierte Konfigurationen
// ----------------------------------------------------------------------------------

// Lade Assets
print('LAI SKRIPT: Lade Assets...');
var trainingSimulationsFc = ee.FeatureCollection(simulationAssetPath);
var regionOfInterest = ee.FeatureCollection(einzugsgebietAssetPath);

// ----------------------------------------------------------------------------------
// Funktionsdefinitionen
// ----------------------------------------------------------------------------------

// Maskiert unerwünschte Pixel (z.B. Wolken, Schatten) basierend auf dem Sentinel-2 Scene Classification Layer (SCL)
function maskS2_SCL(image) {
  var scl = image.select('SCL');
  // Klassen 4 (Vegetation) und 5 (unbewachsener/kahler Boden) werden beibehalten.
  // Diese Auswahl kann je nach Anwendungsfall und gewünschter Landbedeckung angepasst werden.
  var mask = scl.eq(4).or(scl.eq(5));
  return image.updateMask(mask); // Wendet die Maske auf alle Bänder des Bildes an
}

// Bereitet ein Sentinel-2 Bild für die LAI-Vorhersage auf.
// Erzeugt ein Bild mit den standardisierten Bandnamen, die das Random Forest Modell als Eingabe erwartet.
function prepareImageForPrediction(image) {
  // Extrahiert den Solar Zenith Angle (SZA) aus den Metadaten des Bildes und erstellt ein konstantes Bildband.
  var szaImage = ee.Image.constant(image.get('MEAN_SOLAR_ZENITH_ANGLE')).rename('SZA');

  // Wählt die optischen Bänder aus, skaliert sie und benennt sie standardisiert um.
  var opticalBands = image.select(s2OriginalBands)  // Auswahl der definierten Original-S2-Bänder
    .multiply(bandScaling)                       // Anwendung des Skalierungsfaktors
    .rename(s2FeatureBands);                     // Umbenennung in ['blue', 'green', 'red', 'NIR']

  // Kombiniert das SZA-Band mit den optischen Bändern und setzt den Datentyp auf Float.
  return ee.Image.cat(szaImage, opticalBands).toFloat();
}
print('LAI SKRIPT: Funktionsdefinitionen abgeschlossen.');

// ----------------------------------------------------------------------------------
// Random Forest Modell Training
// ----------------------------------------------------------------------------------
print('LAI SKRIPT: Starte Vorbereitung der Trainingsdaten...');

// Transformiert die geladenen Simulationsdaten (Features), um konsistente Property-Namen für das Training sicherzustellen.
// Dies ist notwendig, damit die Spaltennamen aus der CSV-Datei mit den Namen übereinstimmen,
// die das Modell als Eingangsmerkmale (inputProperties) erwartet.
var trainingFeatures = trainingSimulationsFc.map(function(feature) {
  // Erstellt ein neues Dictionary (Objekt) mit den standardisierten Merkmalsnamen und den Werten aus dem aktuellen Feature.
  var properties = {
    'SZA': ee.Number(feature.get(szaPropertyInSimulations)), // Wert für SZA aus der CSV holen
    'blue': ee.Number(feature.get(reflectanceColumnsInSimulations.blue)),   
    'green': ee.Number(feature.get(reflectanceColumnsInSimulations.green)),
    'red': ee.Number(feature.get(reflectanceColumnsInSimulations.red)),
    'NIR': ee.Number(feature.get(reflectanceColumnsInSimulations.NIR)),
    'LAI': ee.Number(feature.get(laiPropertyInSimulations)) 
  };
  return ee.Feature(null, properties); // Erzeugt ein neues Feature (ohne Geometrie) mit den standardisierten Properties.
});
print('LAI SKRIPT - Trainingsdaten transformiert. Anzahl Features:', trainingFeatures.size());

// Definiert und trainiert das Random Forest Regressionsmodell.
print('LAI SKRIPT - Starte Random Forest Training...');
var trainedRfModel = ee.Classifier.smileRandomForest({ // Erstellt ein RF-Modell
    numberOfTrees: numberOfTrees,                       // Anzahl der zu trainierenden Bäume
    seed: randomSeed                                    // Optionaler Seed für Reproduzierbarkeit
  })
  .setOutputMode('REGRESSION') // Wichtig: Teilt dem Modell mit, dass es kontinuierliche Werte vorhersagen soll.
  .train({                     // Startet den Trainingsprozess
    features: trainingFeatures,          // Die vorbereiteten Trainingsdaten (FeatureCollection)
    classProperty: 'LAI',                // Der Name der Property, die die Zielvariable (LAI) enthält
    inputProperties: inputFeatureNames   // Eine Liste der Namen der Properties, die als Eingangsmerkmale dienen sollen
  });
print('LAI SKRIPT - Random Forest Modell Training abgeschlossen.');

// ----------------------------------------------------------------------------------
// Monatliche Verarbeitung und Export der LAI-Vorhersagen
// ----------------------------------------------------------------------------------
print('LAI SKRIPT: Starte Setup für monatliche Verarbeitung und Export...');
print('LAI SKRIPT - Verarbeitungszeitraum: ' + startDate.format('YYYY-MM-dd').getInfo() + ' bis ' + endDate.format('YYYY-MM-dd').getInfo());
print('LAI SKRIPT - Google Drive Export Ordner: ' + outputDriveFolder);

// Erstellt eine Liste von Startdaten für jeden zu verarbeitenden Monat.
var nMonths = endDate.difference(startDate, 'months').round();
var monthList = ee.List.sequence(0, nMonths.subtract(1)).map(function(offset) {
  return startDate.advance(offset, 'month');
});
print('LAI SKRIPT - Anzahl der zu verarbeitenden Monate:', nMonths.getInfo());

// Führt eine client-seitige Schleife über die Monatsliste aus, um für jeden Monat Export-Tasks zu starten.
// .evaluate() holt die Liste der Daten zum Client, forEach iteriert dann darüber.
monthList.evaluate(function(monthsClientList) {
  print('LAI SKRIPT - Callback für monthList.evaluate: Starte Erstellung und Export der monatlichen LAI-Karten...');
  monthsClientList.forEach(function(monthStartClientValue) {
    var monthStart = ee.Date(monthStartClientValue.value); // Startdatum des aktuellen Monats
    var monthEnd = monthStart.advance(1, 'month');         // Enddatum des aktuellen Monats (exklusiv)
    var monthName = monthStart.format('YYYY-MM').getInfo();  // Name des Monats für Benennung und Logs

    print('----------------------------------------------------');
    print('BEGIN PROCESSING MONTH: ' + monthName);
    try {
      // Sentinel-2 Bildersammlung für den aktuellen Monat filtern
      var s2ImageCollection = ee.ImageCollection(s2_collection_id);
      var filteredCollection = s2ImageCollection
        .filterBounds(regionOfInterest) // Filter auf das Untersuchungsgebiet
        .filterDate(monthStart, monthEnd)   // Filter auf den aktuellen Monat
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) // Grober Wolkenfilter (Prozentsatz anpassen)
        .sort('CLOUDY_PIXEL_PERCENTAGE'); // Wählt das am wenigsten bewölkte Bild zuerst

      var collectionSize = filteredCollection.size().getInfo(); // Anzahl der verfügbaren Bilder im Monat
      print('INFO: Anzahl Bilder im Monat ' + monthName + ' (nach Filter): ' + collectionSize);

      if (collectionSize === 0) {
        print('WARNUNG: Keine Bilder für Monat ' + monthName + ' gefunden. Überspringe Monat.');
        return; // Geht zum nächsten Monat in der forEach-Schleife
      }
      var s2Image = ee.Image(filteredCollection.first()); // Nimmt das erste (am wenigsten bewölkte) Bild
      // print('INFO: Verarbeite Bild: ' + s2Image.id().getInfo()); // Zur detaillierten Analyse bei Bedarf aktivieren

      // Bildvorverarbeitung (Maskierung, Merkmalserstellung)
      var maskedS2Image = maskS2_SCL(s2Image);
      var preparedImage = prepareImageForPrediction(maskedS2Image);

      // LAI-Vorhersage mit dem trainierten Random Forest Modell
      // Die .classify()-Methode wendet das Modell pixelweise auf das vorbereitete Bild an.
      // Das Ergebnisband heißt standardmäßig oft 'classification'.
      var laiPredictedImage = preparedImage.classify(trainedRfModel);

      // Ergebnis zuschneiden und Ausgabeband für Klarheit umbenennen
      var clippedLAI = laiPredictedImage.clip(regionOfInterest).rename('LAI');

      // Export des monatlichen LAI-Bildes nach Google Drive
      var exportDescription = 'S2_LAI_RF_' + monthName.replace('-', '_'); // Eindeutiger Name für den Export-Task
      Export.image.toDrive({
        image: clippedLAI,
        description: exportDescription,
        folder: outputDriveFolder,
        scale: exportScale,
        crs: exportCRS,
        region: regionOfInterest.geometry(), // Exportiert die Geometrie des Untersuchungsgebiets
        fileFormat: 'GeoTIFF',
        maxPixels: 1e13 // Maximale Anzahl von Pixeln für den Export
      });
      print('INFO: Export-Task für ' + exportDescription + ' gestartet.');

    } catch (e) {
      // Fängt Fehler ab, die bei der Verarbeitung eines einzelnen Monats auftreten können.
      print('FEHLER bei der Verarbeitung des Monats ' + monthName + ': ' + e.message);
      if (e.stack) { print('Fehler-Stack für Monat ' + monthName + ': ' + e.stack); }
    }
    print('END PROCESSING MONTH: ' + monthName);
  });
  print('LAI SKRIPT - Alle Export-Tasks für die monatliche Verarbeitung wurden (versucht zu) starten.');
});
// --- ENDE Monatliche Verarbeitung ---
print('LAI SKRIPT - Skriptdefinition vollständig beendet.');
