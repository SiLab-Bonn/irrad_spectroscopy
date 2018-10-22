import os
import yaml


# paths
package_path = os.path.dirname(__file__)

examples_path = os.path.join(package_path, 'examples')

testing_path = os.path.join(package_path, 'testing')

static_path = os.path.join(package_path, 'static')

# load files
with open(os.path.join(static_path, 'isotope_lib.yaml'), 'r') as il:
    isotope_lib = yaml.safe_load(il)

with open(os.path.join(static_path, 'element_lib.yaml'), 'r') as el:
    element_lib = yaml.safe_load(el)

with open(os.path.join(static_path, 'xray_coefficients.yaml'), 'r') as xc:
    xray_coeffs = yaml.safe_load(xc)