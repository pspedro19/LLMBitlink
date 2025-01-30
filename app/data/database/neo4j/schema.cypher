// Definición de restricciones e índices
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;
CREATE CONSTRAINT activity_id IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT restaurant_id IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT dish_id IF NOT EXISTS FOR (d:Dish) REQUIRE d.id IS UNIQUE;

// Índices para búsqueda eficiente
CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name);
CREATE INDEX activity_type IF NOT EXISTS FOR (a:Activity) ON (a.type);
CREATE INDEX restaurant_cuisine IF NOT EXISTS FOR (r:Restaurant) ON (r.cuisine_type);
CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date);

// Definición de tipos de nodos
CREATE (l:Location {
    id: 'example_location',
    name: 'Example Location',
    type: 'district',
    description: 'Description text',
    coordinates: point({latitude: 12.1091, longitude: -68.9316}),
    categories: ['historic', 'cultural']
});

CREATE (a:Activity {
    id: 'example_activity',
    name: 'Example Activity',
    type: 'tour',
    description: 'Activity description',
    duration: 120,
    price: 50.00,
    rating: 4.5
});

CREATE (e:Event {
    id: 'example_event',
    name: 'Example Event',
    type: 'cultural',
    description: 'Event description',
    date: date('2024-02-14'),
    duration: 180
});

CREATE (r:Restaurant {
    id: 'example_restaurant',
    name: 'Example Restaurant',
    cuisine_type: 'local',
    price_range: 'moderate',
    rating: 4.2,
    specialties: ['Keshi Yena', 'Stoba']
});

// Definición de relaciones
CREATE (a)-[:LOCATED_IN]->(l);
CREATE (r)-[:LOCATED_IN]->(l);
CREATE (e)-[:TAKES_PLACE_IN]->(l);
CREATE (r)-[:SERVES]->(d:Dish {
    id: 'example_dish',
    name: 'Keshi Yena',
    type: 'main_course',
    price: 25.00
});

// Relaciones de proximidad y rutas
CREATE (l1:Location {id: 'punda'})-[:CONNECTED_TO {
    distance: 0.5,
    walking_time: 10,
    type: 'pedestrian_bridge'
}]->(l2:Location {id: 'otrobanda'});

// Relaciones temáticas y contextuales
// Relaciones temáticas y contextuales
CREATE (a1:Activity {
    id: 'historic_tour',
    name: 'Historic Walking Tour',
    type: 'guided_tour'
})-[:PART_OF]->(t:Theme {
    id: 'cultural_heritage',
    name: 'Cultural Heritage',
    description: 'Explore colonial history and architecture'
});

// Relaciones temporales y estacionales
CREATE (e:Event {
    id: 'carnival',
    name: 'Curaçao Carnival',
    type: 'festival'
})-[:HAPPENS_IN]->(s:Season {
    name: 'Winter',
    months: ['January', 'February', 'March'],
    weather: 'Dry and mild'
});

// Relaciones entre actividades complementarias
CREATE (a1)-[:COMPLEMENTS {
    reason: 'Cultural immersion',
    recommended_sequence: 'after'
}]->(a2:Activity {
    id: 'local_food_tasting',
    name: 'Traditional Food Tour',
    type: 'culinary'
});

// Patrones de visitantes y recomendaciones
CREATE (p:Profile {
    id: 'culture_enthusiast',
    interests: ['history', 'architecture', 'local cuisine'],
    typical_duration: 'half-day',
    budget_range: 'moderate'
})-[:PREFERS]->(a1);

// Rutas y circuitos temáticos
CREATE (r:Route {
    id: 'heritage_trail',
    name: 'Heritage Walking Trail',
    duration: 180,
    difficulty: 'easy',
    highlights: ['Fort Amsterdam', 'Handelskade', 'Mikve Israel Synagogue']
});

// Conectar puntos de interés a la ruta
MATCH (r:Route {id: 'heritage_trail'})
CREATE (poi1:PointOfInterest {
    id: 'fort_amsterdam',
    name: 'Fort Amsterdam',
    type: 'historic_site',
    visit_duration: 45
})-[:PART_OF_ROUTE {sequence: 1}]->(r);

// Información contextual y metadata
CREATE (info:Information {
    type: 'historical_context',
    content: 'Fort Amsterdam was built in 1634...',
    source: 'Curaçao Heritage Foundation',
    last_updated: datetime()
})-[:DESCRIBES]->(poi1);

// Reseñas y valoraciones
CREATE (review:Review {
    id: 'review_001',
    rating: 5,
    text: 'Excellent tour with knowledgeable guide',
    date: date(),
    language: 'English'
})-[:REVIEWS]->(a1);

// Facilidades y servicios
CREATE (f:Facility {
    id: 'visitor_center',
    name: 'Fort Amsterdam Visitor Center',
    type: 'tourist_information',
    services: ['guided tours', 'maps', 'restrooms']
})-[:LOCATED_AT]->(poi1);

// Conexiones de transporte
CREATE (t1:TransportHub {
    id: 'willemstad_bus_terminal',
    name: 'Willemstad Bus Terminal',
    type: 'bus_station'
})-[:CONNECTS_TO {
    mode: 'walking',
    distance: 0.3,
    duration: 5
}]->(poi1);

// Horarios y disponibilidad
CREATE (s:Schedule {
    id: 'fort_schedule',
    opening_hours: {
        weekdays: '9:00-17:00',
        weekends: '10:00-16:00'
    },
    holidays: ['January 1', 'December 25'],
    seasonal_variations: {
        high_season: 'Extended hours available',
        low_season: 'Regular hours'
    }
})-[:APPLIES_TO]->(poi1);

// Costos y tarifas
CREATE (p:Pricing {
    id: 'fort_pricing',
    adult: 10.00,
    child: 5.00,
    student: 7.50,
    senior: 8.00,
    currency: 'USD',
    special_offers: ['group discounts', 'combination tickets']
})-[:APPLIES_TO]->(poi1);

// Requisitos y restricciones
CREATE (r:Requirements {
    id: 'tour_requirements',
    min_age: 8,
    fitness_level: 'moderate',
    accessibility: 'wheelchair friendly',
    required_items: ['comfortable shoes', 'water bottle'],
    recommended_items: ['camera', 'sun protection']
})-[:APPLIES_TO]->(a1);

// Preparación de índices para búsqueda de texto completo
CALL db.index.fulltext.createNodeIndex(
    'locationSearch',
    ['Location'],
    ['name', 'description', 'categories']
);

CALL db.index.fulltext.createNodeIndex(
    'activitySearch',
    ['Activity'],
    ['name', 'description', 'type']
);

// Índices compuestos para búsquedas complejas
CREATE INDEX activity_cost_rating IF NOT EXISTS 
FOR (a:Activity) ON (a.price, a.rating);

CREATE INDEX location_type_coords IF NOT EXISTS 
FOR (l:Location) ON (l.type, l.coordinates);

// Procedimientos almacenados útiles
CREATE PROCEDURE get_nearby_attractions(location_id STRING, max_distance FLOAT)
RETURNS TABLE (
    attraction_id STRING,
    name STRING,
    type STRING,
    distance FLOAT
) 
BEGIN
    MATCH (l:Location {id: location_id})
    MATCH (a:Location)
    WHERE a.id <> location_id
    AND point.distance(l.coordinates, a.coordinates) <= max_distance
    RETURN 
        a.id as attraction_id,
        a.name as name,
        a.type as type,
        point.distance(l.coordinates, a.coordinates) as distance
    ORDER BY distance;
END;