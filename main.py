config = {
    "model_path": None,
    "model_name": "",
    "proj_dir": "",
    "ctx_len": 1024,
    "role" : {"user": {"prefix": [],
                       "post": [65535]},
              "system": {"prefix": [],
                         "post": [65535]},
           "robot": {"prefix": [],
                     "post": [65535]},
              "think": {"prefix": [],
                        "post": [65535]},
           },
    "tokenzier": {"tokenizer_name": ""},
    "trainer": {"": ""},
    "inference": {"init": None,
                  "history" :None,
                  },


}




if __name__ == "__main__":
    pass
