import asyncio
import time
import numpy

expTime = 1
metLevel = 1
nImgRepeat = 3
# 2 degree alpha angle steps
alphaAngles = numpy.arange(0,360,2)
betaAngle = 170

async def sendJaegerCommand(cmdStr):
    cmdID = 3

    reader, writer = await asyncio.open_connection(
        'localhost', 19990)

    while True:
        data = await reader.readline()
        data = data.decode()
        if "yourUserID=" in data:
            userID = int(data.split("yourUserID=")[-1].split(";")[0])
            print("myUserID %i"%userID)
        if "version=" in data:
            # print("break!")
            break

    cmdStr = "%i %s\n"%(cmdID, cmdStr)

    print(f'Send: %s'%cmdStr)
    writer.write(cmdStr.encode())
    await writer.drain()
    while True:
        data = await reader.readline()
        data = data.decode()
        print("read from fvc:", data)
        if "%i %i :"%(userID, cmdID) in data:
            print("command succeeded")
            success = True
            break
        if "%i %i f"%(userID, cmdID) in data:
            print("command failed")
            success = False
            break

    print('Close the connection')
    writer.close()
    await writer.wait_closed()

    return success


async def main():

    for alphaAngle in alphaAngles:
        tstart = time.time()
        moveCmd = "goto -a -f %.2f %.2f"%(alphaAngle, betaAngle)
        success = await sendJaegerCommand(moveCmd)
        if success==False:
            print(moveCmd, "failed exiting")
            return
        for ii in range(nImgRepeat):
            success = await sendJaegerCommand("configuration locad --from-positions --no-write-summary --no-ingest")
            success = await sendJaegerCommand("fvc loop --no-apply --fbi-level %.2f --exposure-time %.2f"%(metLevel, expTime))
        print("angle %.2f took %.2f mins"%(alphaAngle, (time.time()-tstart)/60.))

    lastMove = "jaeger -goto -a -f 180.0 180.0"
    success = await sendJaegerCommand(lastMove)
    if success==False:
        print(moveCmd, "failed exiting")
        return




if __name__ == "__main__":
    asyncio.run(main())
