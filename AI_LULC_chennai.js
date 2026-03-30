// ======================================================
// STEP 1: Define Chennai AOI:EXECUTE THIS IN GOOGLE EARTH ENIGINE
// ======================================================

var districts = ee.FeatureCollection("FAO/GAUL/2015/level2");

var chennai = districts
    .filter(ee.Filter.eq('ADM0_NAME', 'India'))
    .filter(ee.Filter.eq('ADM1_NAME', 'Tamil Nadu'))
    .filter(ee.Filter.eq('ADM2_NAME', 'Chennai'));

Map.centerObject(chennai, 10);
Map.addLayer(chennai, { color: 'red' }, 'Chennai District');


// ======================================================
// STEP 2: Load Sentinel-2 Data
// ======================================================

var imageCollection = ee.ImageCollection("COPERNICUS/S2_SR")
    .filterBounds(chennai)
    .filterDate('2023-01-01', '2023-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .select(['B2', 'B3', 'B4', 'B8', 'B11']);  // Added B11 for NDBI

var image = imageCollection.median().clip(chennai);


// ======================================================
// STEP 3: Display RGB
// ======================================================

Map.addLayer(image, {
    bands: ['B4', 'B3', 'B2'],
    min: 0,
    max: 3000
}, 'Sentinel-2 RGB');


// ======================================================
// STEP 4: Calculate NDVI
// ======================================================

var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

Map.addLayer(ndvi, {
    min: -1,
    max: 1,
    palette: ['blue', 'white', 'green']
}, 'NDVI');


// ======================================================
// STEP 5: Calculate NDBI (Urban Index)
// ======================================================

var ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI');


// ======================================================
// STEP 6: Add Indices to Image
// ======================================================

var imageWithIndices = image
    .addBands(ndvi)
    .addBands(ndbi);


// ======================================================
// STEP 7: Convert Geometries to Features
// ======================================================

var waterF = ee.Feature(water).set('class', 0);
var vegF = ee.Feature(veg).set('class', 1);
var urbanF = ee.Feature(urban).set('class', 2);
var barrenF = ee.Feature(barren).set('class', 3);


// ======================================================
// STEP 8: Combine Training Polygons
// ======================================================

var trainingPolygons = ee.FeatureCollection([
    waterF,
    vegF,
    urbanF,
    barrenF
]);


// ======================================================
// STEP 9: Extract Training Data
// ======================================================

var training = imageWithIndices.sampleRegions({
    collection: trainingPolygons,
    properties: ['class'],
    scale: 10
});


// ======================================================
// STEP 10: Split Training & Testing Data
// ======================================================

var withRandom = training.randomColumn('random');

var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.7));


// ======================================================
// STEP 11: Train Random Forest
// ======================================================

var classifier = ee.Classifier.smileRandomForest(100).train({
    features: trainSet,
    classProperty: 'class',
    inputProperties: imageWithIndices.bandNames()
});


// ======================================================
// STEP 12: Classify Image
// ======================================================

var classified = imageWithIndices.classify(classifier);

Map.addLayer(classified, {
    min: 0,
    max: 3,
    palette: ['blue', 'green', 'gray', 'yellow']
}, 'Final LULC Classification');


// ======================================================
// STEP 13: Accuracy Assessment
// ======================================================

var testClassification = testSet.classify(classifier);

var confusionMatrix = testClassification.errorMatrix('class', 'classification');

print('Confusion Matrix:', confusionMatrix);
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa Coefficient:', confusionMatrix.kappa());
// ======================================================
// STEP 14: Calculate Area of Each Class
// ======================================================

// Calculate pixel area
var areaImage = ee.Image.pixelArea().addBands(classified);

// Group area by class
var areaStats = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'class',
    }),
    geometry: chennai,
    scale: 10,
    maxPixels: 1e13
});

print('Area by Class (sq meters):', areaStats);

// Convert to square kilometers
var areas = ee.List(areaStats.get('groups'));

var areaSqKm = areas.map(function (item) {
    var dict = ee.Dictionary(item);
    return ee.Dictionary({
        class: dict.get('class'),
        area_sq_km: ee.Number(dict.get('sum')).divide(1e6)
    });
});

print('Area by Class (sq km):', areaSqKm);
Export.image.toDrive({
    image: classified,
    description: 'Chennai_LULC',
    folder: 'GEE_Exports',
    fileNamePrefix: 'Chennai_LULC',
    region: chennai,
    scale: 10,
    maxPixels: 1e13
});