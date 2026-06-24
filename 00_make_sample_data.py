# -*- coding: utf-8 -*-
"""
00_make_sample_data.py
-----------------------
Yelp Academic Dataset (business) seyrek/buyuk oldugu icin, calismayi uctan uca
tekrarlanabilir kilmak adina ayni semaya (name + categories + city + stars ...)
uyan temsili bir ornek isletme JSON'u uretir.

Gercek 'yelp_academic_dataset_business.json' dosyaniz elinizdeyse bu adimi
atlayabilir, 01_preprocess.py icindeki SOURCE_JSON yolunu gercek dosyaya
cevirebilirsiniz. Sema ayni oldugu icin tum boru hatti degismeden calisir.

Cikti: data/yelp_academic_dataset_business.json  (her satir bir JSON kaydi)
"""
import os
import json
import random

random.seed(2024)  # tekrarlanabilirlik

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Her aile: (kategori havuzu, isim onekleri, isim cekirdekleri)
FAMILIES = {
    "coffee": {
        "categories": ["Coffee & Tea", "Cafes", "Bakeries", "Breakfast & Brunch",
                       "Donuts", "Juice Bars & Smoothies", "Tea Rooms"],
        "prefix": ["Morning", "Daily", "Urban", "Corner", "Bean", "Sunrise", "Cozy", "Roasted"],
        "core": ["Coffee", "Cafe", "Roasters", "Espresso", "Brew House", "Beans", "Bakery"],
    },
    "pizza": {
        "categories": ["Restaurants", "Pizza", "Italian", "Pasta", "Wine Bars", "Calzones"],
        "prefix": ["Bella", "Nonna", "Little", "Original", "Famous", "Mama", "Rustic"],
        "core": ["Pizza", "Pizzeria", "Trattoria", "Italian Kitchen", "Pasta House", "Ristorante"],
    },
    "mexican": {
        "categories": ["Restaurants", "Mexican", "Tacos", "Tex-Mex", "Burritos", "Latin American"],
        "prefix": ["El", "La", "Casa", "Taco", "Fiesta", "Agave", "Sol"],
        "core": ["Taqueria", "Cantina", "Grill", "Tacos", "Burrito Bar", "Cocina"],
    },
    "asian": {
        "categories": ["Restaurants", "Chinese", "Japanese", "Sushi Bars", "Thai", "Ramen", "Noodles", "Asian Fusion"],
        "prefix": ["Golden", "Lucky", "Tokyo", "Bamboo", "Red", "Jade", "Sakura"],
        "core": ["Sushi", "Ramen", "Noodle House", "Wok", "Dragon", "Kitchen", "Bistro"],
    },
    "burgers": {
        "categories": ["Restaurants", "Burgers", "Fast Food", "American (Traditional)", "Sandwiches", "Hot Dogs", "Fries"],
        "prefix": ["Big", "Smash", "Classic", "Hungry", "Route", "Downtown", "Prime"],
        "core": ["Burgers", "Burger Joint", "Grill", "Diner", "Sandwich Shop", "Eatery"],
    },
    "bars": {
        "categories": ["Bars", "Nightlife", "Cocktail Bars", "Pubs", "Breweries", "Beer", "Sports Bars", "Lounges"],
        "prefix": ["The", "Old", "Iron", "Copper", "Blue", "Tap", "Hop"],
        "core": ["Tavern", "Pub", "Brewery", "Cocktail Bar", "Alehouse", "Lounge", "Saloon"],
    },
    "beauty": {
        "categories": ["Beauty & Spas", "Hair Salons", "Nail Salons", "Day Spas", "Barbers", "Skin Care", "Waxing"],
        "prefix": ["Glow", "Pure", "Luxe", "Bliss", "Polished", "Serene", "Radiant"],
        "core": ["Salon", "Spa", "Nails", "Hair Studio", "Barbershop", "Beauty Bar"],
    },
    "auto": {
        "categories": ["Automotive", "Auto Repair", "Tires", "Oil Change Stations", "Car Wash", "Body Shops", "Auto Parts"],
        "prefix": ["Pro", "Express", "Precision", "All-Star", "Highway", "Quick", "Master"],
        "core": ["Auto Repair", "Tire Center", "Car Wash", "Auto Body", "Motors", "Garage", "Service Center"],
    },
    "health": {
        "categories": ["Health & Medical", "Dentists", "Doctors", "Chiropractors", "Physical Therapy", "Optometrists"],
        "prefix": ["Family", "Modern", "Bright", "Gentle", "Community", "Advanced", "Lakeside"],
        "core": ["Dental", "Dentistry", "Family Clinic", "Medical Center", "Chiropractic", "Health Group"],
    },
    "fitness": {
        "categories": ["Active Life", "Gyms", "Yoga", "Fitness & Instruction", "Trainers", "Pilates", "Cycling Classes"],
        "prefix": ["Peak", "Iron", "Flow", "Core", "Summit", "Pulse", "Elevate"],
        "core": ["Fitness", "Gym", "Yoga Studio", "Crossfit", "Wellness", "Training Lab"],
    },
    "grocery": {
        "categories": ["Shopping", "Grocery", "Convenience Stores", "Organic Stores", "Farmers Market", "Specialty Food"],
        "prefix": ["Fresh", "Green", "Market", "Family", "Harvest", "Nature", "Daily"],
        "core": ["Grocery", "Market", "Foods", "Mart", "Provisions", "Pantry"],
    },
    "home": {
        "categories": ["Home Services", "Plumbing", "Electricians", "Contractors", "Heating & Air Conditioning/HVAC", "Roofing"],
        "prefix": ["Reliable", "Premier", "All", "Trusted", "Hometown", "Apex", "Guardian"],
        "core": ["Plumbing", "Electric", "HVAC", "Contractors", "Home Services", "Heating & Cooling"],
    },
    "pets": {
        "categories": ["Pets", "Pet Stores", "Veterinarians", "Pet Groomers", "Animal Shelters", "Dog Walkers"],
        "prefix": ["Happy", "Furry", "Paws", "Loyal", "Whiskers", "Wag", "Cozy"],
        "core": ["Pet Shop", "Veterinary Clinic", "Pet Grooming", "Animal Hospital", "Pet Care"],
    },
    "hotels": {
        "categories": ["Hotels & Travel", "Hotels", "Event Planning & Services", "Resorts", "Bed & Breakfast"],
        "prefix": ["Grand", "Royal", "Park", "Lakeview", "Plaza", "Garden", "Heritage"],
        "core": ["Hotel", "Inn", "Suites", "Resort", "Lodge", "Bed & Breakfast"],
    },
}

CITIES = ["Phoenix", "Tampa", "Austin", "Nashville", "Portland", "Boulder",
          "Reno", "Tucson", "Columbus", "Madison"]
STATES = {"Phoenix": "AZ", "Tampa": "FL", "Austin": "TX", "Nashville": "TN",
          "Portland": "OR", "Boulder": "CO", "Reno": "NV", "Tucson": "AZ",
          "Columbus": "OH", "Madison": "WI"}


def make_name(fam):
    return "{} {}".format(random.choice(fam["prefix"]), random.choice(fam["core"]))


def make_categories(fam):
    cats = fam["categories"]
    k = random.randint(2, min(4, len(cats)))
    chosen = random.sample(cats, k)
    return ", ".join(chosen)


def main():
    n_per_family = 55
    records = []
    bid = 1000
    for key, fam in FAMILIES.items():
        for _ in range(n_per_family):
            bid += 1
            city = random.choice(CITIES)
            rec = {
                "business_id": "b{}".format(bid),
                "name": make_name(fam),
                "city": city,
                "state": STATES[city],
                "stars": round(random.choice([3.0, 3.5, 4.0, 4.0, 4.5, 4.5, 5.0]), 1),
                "review_count": random.randint(5, 850),
                "is_open": random.choice([0, 1, 1, 1]),
                "categories": make_categories(fam),
            }
            records.append(rec)

    random.shuffle(records)
    out_path = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Uretildi: {} ({} isletme kaydi, {} aile)".format(
        out_path, len(records), len(FAMILIES)))


if __name__ == "__main__":
    main()
