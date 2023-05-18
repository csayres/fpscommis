import random

class ScriptClass(object):

    def __init__(self, sr):
        pass

    def run(self, sr):
        """
        Take a series of repeat exposures with the FVC
        at different rotator angles
        """
        alt = 70
        az = 121
        rotTests = [-35, 175, 180, 190, 40, 265]
        fvcRepeat = 11
        wholeRepeat = 6

        for ii in range(wholeRepeat):
            random.shuffle(rotTests)
            sr.showMsg("On iter %i of %i"% (ii+1, wholeRepeat))
            for cmdRot in rotTests:
                tccCmd = "track %.2f, %.2f mount /rota=%.5f /rottype=mount"%(az, alt, cmdRot)

                # command the slew
                yield sr.waitCmd(
                    actor = "tcc",
                    cmdStr = tccCmd
                    )

                # command axis stop
                yield sr.waitCmd(
                    actor = "tcc",
                    cmdStr = "axis stop"
                    )

                # wait for 5 seconds
                yield sr.waitMS(5 * 1000)

                # begin fvc iter
                for ii in range(fvcRepeat):
                    yield sr.waitCmd(
                        actor = "jaeger",
                        cmdStr = "fvc loop --no-apply --no-write-summary",
                        checkFail=False
                        )

    def end(self, sr):
        pass


