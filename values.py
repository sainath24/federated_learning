TOKEN_BUFFER_SIZE = 16
BUFFER_SIZE = 4096
SEPARATOR = "&"
TOKEN_REGEX = "[a-f][0-9]{16}"

# CLIENT STATUS
NEW_CLIENT = 0
GLOBAL_MODEL_SENT = 1
LOCAL_MODEL_RECEIVED = 2

# RESPONSES
metadata_valid = "METADATA_OK"
metadata_invalid = "METADATA_INVALID"

# SERVER RESPONSES
no_update_available = "NO_UPDATE"
receive_model_success = "RECEIVE_OK"
receive_model_fail = "RECEIVE_FAIL"
receive_token_valid = "TOKEN_OK"
receive_token_invalid = "TOKEN_INVALID"
send_token_fail = "SEND_TOKEN_FAIL"
client_invalid_response = "INVLAID_RESPONSE"
send_model_fail = "SEND_MODEL_FAIL"


#CLIENT
REQUIRES_TOKEN = "0000000000000000"
client_connection_success = "CONNECTED TO SERVER"
client_connection_fail_retry = "CONNECTION TO SERVER FAILED, RETRYING..."
client_connection_fail = "CONNECTION TO SERVER FAILED"
client_invalid_token = "INVALID TOKEN, REQUESTING NEW TOKEN"
client_receive_model_success = "MODEL_RECEIVED"
get_failed_retry = "UNABLE TO RECEIVE MODEL, RETRYING..."
get_failed = "UNABLE TO RECEIVE MODEL"
send_failed_retry = "UNABLE TO SEND MODEL, RETRYING..."
send_failed = "UNABLE TO SEND MODEL"
