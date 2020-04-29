import re
import os
import sys


def get_llvm_target():
    # Get host information from llc
    stream = os.popen('llc --version')
    cpu_info = stream.read()

    # Parse out cpu line
    cpu = re.search('(?<=Host CPU: ).+', cpu_info).group(0)

    # Next extract attribute string.
    platform = sys.platform
    # Linux
    if platform == 'linux' or platform == 'linux2':
        stream = os.popen('lscpu')
        feature_info = stream.read()
        features = re.search('(?<=Flags: ).+', feature_info).group(0)
        features = features.lower().strip().split(' ')
        stream = os.popen('nproc')
        core_info = stream.read()
        cores = core_info.lower().strip()
    # Mac
    elif platform == 'darwin':
        stream = os.popen('sysctl -a | grep machdep.cpu')
        feature_info = stream.read()
        features = re.search('(?<=machdep.cpu.features: ).+', feature_info).group(0)
        features = features.lower().split(' ')
        cores = re.search('(?<=machdep.cpu.thread_count: ).+', feature_info).group(0)
        cores = cores.lower().split()

    else:
        raise ValueError("Platform %s is not supported." % platform)
    attrs = ''
    for f in features:
        attrs += '+%s,' % f
    # Remove final comma
    attrs = attrs[:-1]

    # Finally extract the target triple. Note that although not needed its
    # nice to have for completion.
    host_target = os.popen('llvm-config --host-target').read().strip('\n')

    # Compose target string.
    target = 'llvm -mcpu=%s -mattr=%s -target=%s' % (cpu, attrs, host_target)
    return target, int(cores)