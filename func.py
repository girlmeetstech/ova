#
# oci-load-file-into-adw-python version 1.0.
#
# Copyright (c) 2020 Oracle, Inc.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl.
#

import io
import json
import pandas
import requests
import base64
import csv
from io import StringIO
from fdk import response

import oci
from vision_service_python_client.ai_service_vision_client import AIServiceVisionClient
from vision_service_python_client.models.analyze_image_details import AnalyzeImageDetails
from vision_service_python_client.models.image_object_detection_feature import ImageObjectDetectionFeature
from vision_service_python_client.models.inline_image_details import InlineImageDetails

def soda_insert(ordsbaseurl, schema, dbuser, dbpwd, document):
    auth=(dbuser, dbpwd)
    sodaurl = ordsbaseurl + schema + '/soda/latest/'
    collectionurl = sodaurl + "foodcount"
    headers = {'Content-Type': 'application/json'}
    r = requests.post(collectionurl, auth=auth, headers=headers, data=json.dumps(document))
    r_json = {}
    try:
        r_json = json.loads(r.text)
    except ValueError as e:
        print(r.text, flush=True)
        raise
    return r_json


def load_data(signer, namespace, bucket_name, object_name, ordsbaseurl, schema, dbuser, dbpwd):
    client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    try:
        print("INFO - About to read object {0} in bucket {1}...".format(object_name, bucket_name), flush=True)
        # we assume the file can fit in memory, otherwise we have to use the "range" argument and loop through the file
        image = client.get_object(namespace, bucket_name, object_name)
        load_json = do(signer,image)
        insert_status = soda_insert(ordsbaseurl, schema, dbuser, dbpwd, row)
            if "id" in insert_status["items"][0]:
                print("INFO - Successfully inserted document ID " + insert_status["items"][0]["id"], flush=True)
            else:
                raise SystemExit("Error while inserting: " + insert_status)    
        
def vision(dip, txt):
    encoded_string = base64.b64encode(requests.get(txt).content)
    image_object_detection_feature = ImageObjectDetectionFeature()
    image_object_detection_feature.max_results = 5
    features = [image_object_detection_feature]
    analyze_image_details = AnalyzeImageDetails()
    inline_image_details = InlineImageDetails()
    inline_image_details.data = encoded_string.decode('utf-8')
    analyze_image_details.image = inline_image_details
    analyze_image_details.features = features
    le = dip.analyze_image(analyze_image_details=analyze_image_details)
    if len(le.data.image_objects) > 0:
      return json.loads(le.data.image_objects.__repr__())
    return ""

        
        if image.status == 200:
            print("INFO - Object {0} is read".format(object_name), flush=True)
            input_image = str(image.data)
            reader = image.DictReader(input_image.split('\n'), delimiter=',')
            for row in reader:
                print("INFO - inserting:")
                print("INFO - " + json.dumps(row), flush=True)
                insert_status = soda_insert(ordsbaseurl, schema, dbuser, dbpwd, row)
                if "id" in insert_status["items"][0]:
                    print("INFO - Successfully inserted document ID " + insert_status["items"][0]["id"], flush=True)
                else:
                    raise SystemExit("Error while inserting: " + insert_status)
        else:
            raise SystemExit("cannot retrieve the object" + str(object_name))
    except Exception as e:
        raise SystemExit(str(e))
    print("INFO - All documents are successfully loaded into the database", flush=True)


def move_object(signer, namespace, source_bucket, destination_bucket, object_name):
    objstore = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    objstore_composite_ops = oci.object_storage.ObjectStorageClientCompositeOperations(objstore)
    resp = objstore_composite_ops.copy_object_and_wait_for_state(
        namespace, 
        source_bucket, 
        oci.object_storage.models.CopyObjectDetails(
            destination_bucket=destination_bucket, 
            destination_namespace=namespace,
            destination_object_name=object_name,
            destination_region=signer.region,
            source_object_name=object_name
            ),
        wait_for_states=[
            oci.object_storage.models.WorkRequest.STATUS_COMPLETED,
            oci.object_storage.models.WorkRequest.STATUS_FAILED])
    if resp.data.status != "COMPLETED":
        raise Exception("cannot copy object {0} to bucket {1}".format(object_name,destination_bucket))
    else:
        resp = objstore.delete_object(namespace, source_bucket, object_name)
        print("INFO - Object {0} moved to Bucket {1}".format(object_name,destination_bucket), flush=True)


def handler(ctx, data: io.BytesIO=None):
    signer = oci.auth.signers.get_resource_principals_signer()
    object_name = bucket_name = namespace = ordsbaseurl = schema = dbuser = dbpwd = ""
    try:
        cfg = ctx.Config()
        input_bucket = cfg["input-bucket"]
        processed_bucket = cfg["processed-bucket"]
        ordsbaseurl = cfg["ords-base-url"]
        schema = cfg["db-schema"]
        dbuser = cfg["db-user"]
        dbpwd = cfg["dbpwd-cipher"]
    except Exception as e:
        print('Missing function parameters: bucket_name, ordsbaseurl, schema, dbuser, dbpwd', flush=True)
        raise
    try:
        body = json.loads(data.getvalue())
        print("INFO - Event ID {} received".format(body["eventID"]), flush=True)
        print("INFO - Object name: " + body["data"]["resourceName"], flush=True)
        object_name = body["data"]["resourceName"]
        print("INFO - Bucket name: " + body["data"]["additionalDetails"]["bucketName"], flush=True)
        if body["data"]["additionalDetails"]["bucketName"] != input_bucket:
            raise ValueError("Event Bucket name error")
        print("INFO - Namespace: " + body["data"]["additionalDetails"]["namespace"], flush=True)
        namespace = body["data"]["additionalDetails"]["namespace"]
    except Exception as e:
        print('ERROR: bad Event!', flush=True)
        raise
    load_data(signer, namespace, input_bucket, object_name, ordsbaseurl, schema, dbuser, dbpwd)
    move_object(signer, namespace, input_bucket, processed_bucket, object_name)

    return response.Response(
        ctx, 
        response_data=json.dumps({"status": "Success"}),
        headers={"Content-Type": "application/json"}
    )


def do(signer, data):
    dip = AIServiceVisionClient(config={}, signer=signer)
    body = json.loads(data.getvalue())
    input_parameters = body.get("parameters")
    col = input_parameters.get("column")
    input_data = base64.b64decode(body.get("data")).decode()
    df = pandas.read_json(StringIO(input_data), lines=True)
    df['enr'] = df.apply(lambda row : vision(dip,row[col]), axis = 1)
    #Explode the array of aspects into row per entity
    dfe = df.explode('enr', True)
    #Add a column for each property we want to return from image_objects struct
    ret = pandas.concat([dfe,pandas.DataFrame((d for idx, d in dfe['enr'].iteritems()))], axis = 1)
    #Drop array of aspects column
    ret = ret.drop(['enr'],axis = 1)
    #Drop the input text column we don't need to return that (there may be other columns there)
    ret = ret.drop([col],axis = 1)
    print(ret.columns)
    for i in range(4):
        ret['x' + str(i)] = ret.apply(lambda row: row['bounding_polygon']['normalized_vertices'][i]['x'], axis = 1)
        ret['y' + str(i)] = ret.apply(lambda row: row['bounding_polygon']['normalized_vertices'][i]['y'], axis = 1)
    ret = ret.drop(['bounding_polygon'],axis = 1)
    res = ret.to_json(orient = 'records')
    return res
