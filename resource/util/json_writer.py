# -*- coding: utf-8 -*-

import json
import time
import logging

jw_logger = logging.getLogger("main.json_writer")


class JsonWriter:
    def __init__(self, gth_key=None, hyp_key=None):
        super(JsonWriter, self).__init__()
        self.gth_key = gth_key or "gth"
        self.hyp_key = hyp_key or "hyp"
        self.identity_key = "identity"

    def write2file(self, filename, gths, hyps,identites=None):
        lens = len(gths)
        context = []
        for i in range(lens):
            sample = dict()
            if identites is not None:
                sample[self.identity_key] = identites[i]
            sample[self.gth_key] = gths[i]
            sample[self.hyp_key] = hyps[i]
            context.append(sample)

        jw_logger.info("dumping to {} ".format(filename))
        s = time.time()
        json.dump(context, open(filename, "w"), ensure_ascii=False, indent=4)
        jw_logger.info("dump done in {} seconds".format(time.time() - s))
