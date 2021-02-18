import secrets
import re

import sys
sys.path.append("../")
import values

def gen_token(tokens):  # GENERATE NEW TOKEN
    try:
        # GENERATE NEW TOKEN
        token = secrets.token_hex(8)
        while token in tokens.keys():
            token = secrets.token_hex(8)
        return token
    except Exception as e:
        print("\nEXCEPTION IN send_token: ", e)
    return False

def check_valid_client(token, client_data):
    if token in client_data.keys():
        return True
    return False

def check_token_validity(token, client_data):
    check = re.fullmatch(values.TOKEN_REGEX, token)
    if check and check_valid_client(token, client_data):  # VALID HEXADECIMAL TOKEN
        return True
    return False

def check_token_in_list(token, token_list):
    if token in token_list.keys():
        return True
    else:
        return False
