import sys
import json
import datetime
import logging

import torch
import torch.nn.functional as F

from common import args, evaluate

import urllib.parse
import urllib.request

# Graph API Specific Information
GRAPH_CONFIG = {
    "ACCESS_TOKEN": "129645254407984|e3a11a7986b43b340bb74394ca0abf4f",
    "PUBLISH_ENDPOINT": "https://graph.facebook.com/v3.1/herophilus_publish",
}

def publish_model(model, args, owner_name, comment, script_name):
    """
    Publishes the model to the leaderboard. 

    :param model: The model to evaluate. The model should be callable with a batch of masked k-space images
                  and masks as inputs and return a batch of predicted images.
    :param args:
    :param owner_name: Github handle of model owner.
    :param comment: Model builder's free-text description of the model.
    :param script_name: Calling script name of the trained model.

    :return: Network response from the POST request.
    """
        
    logging.info("Publishing model from script {} by {}, comment: {}\n description: {}".format(script_name, owner_name, comment, str(model)))
        
    logging.info("Calculating training validation metrics...")
    validation_metrics = evaluate.evaluate_model('val', model, args)

    logging.info("Calculating public leaderboard validation metrics...")   
    public_leaderboard_metrics = evaluate.evaluate_model('public_leaderboard', model, args)        

    payload = {
        "access_token": GRAPH_CONFIG["ACCESS_TOKEN"],
        "commit_hash": "",
        "cron_job_date": int(datetime.datetime.now().timestamp()),
        "comment": comment,
        "script_name": script_name,
        "model_owner": owner_name,
        "model_description": str(model),
        "time_uploaded": int(datetime.datetime.now().timestamp()),
        "args_ran": str(args),
        "training_validation_metrics": json.dumps(validation_metrics),
        "public_leaderboard_metrics": json.dumps(public_leaderboard_metrics),
    }
    data = urllib.parse.urlencode(payload)
    data = data.encode('ascii') # data should be bytes    
    logging.info("Payload built: {}".format(data))
    
    logging.info("Making POST request to {}".format(GRAPH_CONFIG['PUBLISH_ENDPOINT']))
    req = urllib.request.Request(
        GRAPH_CONFIG['PUBLISH_ENDPOINT'],
        data,
        headers={
            'Content-type': 'application/json',
            'Accept': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        the_page = response.read()
        logging.info("response: {}".format(str(the_page))) 
        return True
    return False
