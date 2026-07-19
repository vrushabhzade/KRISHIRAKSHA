"""DynamoDB repository. A local repository keeps the app usable without AWS credentials."""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
import boto3
from boto3.dynamodb.conditions import Key
from .config import settings

def now() -> str: return datetime.now(timezone.utc).isoformat()
def clean(value):
    if isinstance(value, float): return Decimal(str(value))
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value
def jsonify(value):
    if isinstance(value, Decimal): return float(value)
    if isinstance(value, dict): return {k:jsonify(v) for k,v in value.items()}
    if isinstance(value, list): return [jsonify(v) for v in value]
    return value

class Repository:
    def __init__(self): self.local: dict[str,list[dict]]={}
    @property
    def table(self): return boto3.resource("dynamodb",region_name=settings.aws_region).Table(settings.dynamodb_table)
    def save_farmer(self, farmer:dict):
        item={**farmer,"PK":farmer["farmerId"],"SK":"FARMER","entityType":"Farmer"}
        if settings.use_dynamodb: self.table.put_item(Item=clean(item))
        else: self.local.setdefault(farmer["farmerId"],[]); self.local[farmer["farmerId"]]=[item]+[x for x in self.local[farmer["farmerId"]] if x["SK"]!="FARMER"]
        return item
    def farmer(self, farmer_id:str):
        if settings.use_dynamodb: return jsonify(self.table.get_item(Key={"PK":farmer_id,"SK":"FARMER"}).get("Item"))
        return next((x for x in self.local.get(farmer_id,[]) if x["SK"]=="FARMER"),None)
    def all_farmers(self):
        if settings.use_dynamodb:
            items=[]; args={"FilterExpression":"entityType = :v","ExpressionAttributeValues":{":v":"Farmer"}}
            while True:
                response=self.table.scan(**args); items.extend(response.get("Items",[]))
                if "LastEvaluatedKey" not in response: return jsonify(items)
                args["ExclusiveStartKey"]=response["LastEvaluatedKey"]
        return [x for values in self.local.values() for x in values if x["SK"]=="FARMER"]
    def event(self, farmer_id:str, event_type:str, data:dict):
        timestamp=now(); item={**data,"PK":farmer_id,"SK":timestamp,"entityType":event_type,"timestamp":timestamp}
        if settings.use_dynamodb: self.table.put_item(Item=clean(item))
        else: self.local.setdefault(farmer_id,[]).append(item)
        return jsonify(item)
    def scans(self, farmer_id:str):
        if settings.use_dynamodb: items=self.table.query(KeyConditionExpression=Key("PK").eq(farmer_id)).get("Items",[])
        else: items=self.local.get(farmer_id,[])
        return sorted([jsonify(x) for x in items if x.get("entityType")=="ScanHistory"],key=lambda x:x["timestamp"],reverse=True)

repo=Repository()
