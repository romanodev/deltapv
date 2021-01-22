import psc
import logging

logger = logging.getLogger()
logger.setLevel("INFO")
logger.addHandler(logging.FileHandler("randomsearch.log"))

if __name__ == "__main__":

    for niter in range(100):
        x = psc.sample()
        try:
            obj = psc.f(x)
            logger.info(f"Sample: {list(x)}; Status: Success; Objective: {obj}")
        except Exception:
            logger.error("Failed to converge!")
            logger.info(f"Sample: {list(x)}; Status: Failure; Objective: None")
