import io
import json
import pandas
import requests
import base64
from io import StringIO
from fdk import response

import oci
from vision_service_python_client.ai_service_vision_client import AIServiceVisionClient
from vision_service_python_client.models.analyze_image_details import AnalyzeImageDetails
from vision_service_python_client.models.image_object_detection_feature import ImageObjectDetectionFeature
from vision_service_python_client.models.inline_image_details import InlineImageDetails

def handler(ctx, data: io.BytesIO=None):
    signer = oci.auth.signers.get_resource_principals_signer()
    resp = do(signer,data)
    return response.Response(
        ctx, response_data=resp,
        headers={"Content-Type": "application/json"}
    )
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
