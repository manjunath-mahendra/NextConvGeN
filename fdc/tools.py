import time


def count(testFn, items):
    s = 0
    for x in items:
        if testFn(x):
            s += 1
    return s


def indent(text, i="  "):
  result = ""
  for x in text.split("\n"):
    result += i + x + "\n"
  return result



def indentPair(a, b, e="", i="  "):
  m = a + " "
  if len(m) < 32:
    m += "_" * (32 - len(m))
  if len(b) < 16:
    m += "_" * (10 - len(b))
  m += " "
  m += b
  if e == False:
    pass
  elif e == True:
    m += " *"
  else:
    m += e
  print("  " + m)


class Timing:
    def __init__(self, name="Duration"):
        self.name = name
        self.tStart = time.process_time()
        self.tStepStart = self.tStart

    def step(self, message=""):
        now = time.process_time()
        duration = now - self.tStart
        durationStep = now - self.tStepStart
        self.tStepStart = now

        if message == "":
            print(f"{self.name}: {durationStep:0.5f} / {duration:0.3f}s")
        else:
            print(f"{self.name} ({message}): {durationStep:0.5f} / {duration:0.3f}s")
        return duration
