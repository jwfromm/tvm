import re
import sys
import subprocess


def get_llvm_target():
    # Get host information from llc
    stream = subprocess.Popen('llc --version', shell=True, stdout=subprocess.PIPE)
    cpu_info = stream.stdout.read().decode('utf-8')

    # Parse out cpu line
    cpu = re.search('(?<=Host CPU: ).+', cpu_info).group(0)

    # Next extract attribute string.
    platform = sys.platform
    # Linux
    if platform == 'linux' or platform == 'linux2':
        stream = subprocess.Popen('lscpu', shell=True, stdout=subprocess.PIPE)
        feature_info = stream.stdout.read().decode('utf-8')
        features = re.search('(?<=Flags: ).+', feature_info).group(0)
        features = features.lower().strip().split(' ')
        stream = subprocess.Popen('nproc', shell=True, stdout=subprocess.PIPE)
        core_info = stream.stdout.read().decode('utf-8')
        cores = core_info.lower().strip()
        march = re.search('(?<=Architecture: ).+', feature_info).group(0).strip()
        # Special case for x86_64 mismatch between underscore and hyphen
        if march == 'x86_64':
            march = 'x86-64'
    # Mac
    elif platform == 'darwin':
        stream = subprocess.Popen(
            'sysctl -a | grep machdep.cpu', shell=True, stdout=subprocess.PIPE)
        feature_info = stream.stdout.read().decode('utf-8')
        features = re.search('(?<=machdep.cpu.features: ).+', feature_info).group(0)
        features = features.lower().split(' ')
        cores = re.search('(?<=machdep.cpu.thread_count: ).+', feature_info).group(0)
        cores = cores.lower().split()
        march = re.search('?<=machdep.cpu.brand_string: ).+', feature_info).group(0)
        # Currently only two possibilties for recent macs.
        if "Intel" in march:
            march = 'x86-64'
        else:
            march = 'arm64'

    else:
        raise ValueError("Platform %s is not supported." % platform)

    # Now we'll extract the architecture of the target.
    stream = subprocess.Popen('llc --version', shell=True, stdout=subprocess.PIPE)
    march_info = stream.stdout.read().decode('utf-8')
    # Remove header.
    march_options = re.search('(?<=Registered Targets:).*', march_info, re.DOTALL).group(0)
    march_list = []
    for march_line in march_options.split('\n'):
        if march_line != '':
            march_list.append(march_line.strip().split(' ')[0])

    valid_march = False
    if march in march_list:
        valid_march = True

    # Build the base target.
    host_target = subprocess.Popen(
        'llvm-config --host-target', shell=True,
        stdout=subprocess.PIPE).stdout.read().decode('utf-8').strip('\n')
    target = 'llvm -mcpu=%s -target=%s' % (cpu, host_target)

    # If possible, add more attribute information.
    if valid_march:
        # Get list of valid attributes for the target architecture.
        sp = subprocess.Popen(
            'llc -march=%s -mattr=help' % march, shell=True, stderr=subprocess.PIPE)
        attrs_info = sp.stderr.read().decode('utf-8')
        supported_attrs = re.search(
            '(?<=Available features for this target:).*(?=Use \+feature to enable a feature)',
            attrs_info, re.DOTALL).group(0)
        attrs_list = []
        for attrs_line in supported_attrs.split('\n'):
            if attrs_line != '':
                attrs_list.append(attrs_line.strip().split(' ')[0])

        attrs = []
        # Find which features are supported attrs.
        for f in features:
            if f in attrs_list:
                attrs.append(f)

        # Compuse attributes into valid string.
        attrs_string = ''
        for attr in attrs:
            attrs_string += '+%s,' % attr
        # Remove final comma
        attrs_string = attrs_string[:-1]

        # Now we can add more information to the llvm target.
        target = "%s -mattr=%s" % (target, attrs_string)

    return target, int(cores)


if __name__ == "__main__":
    print(get_llvm_target())