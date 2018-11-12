import os
import yaml


# paths
package_path = os.path.dirname(__file__)

examples_path = os.path.join(package_path, 'examples')

testing_path = os.path.join(package_path, 'testing')

tables_path = os.path.join(package_path, 'tables')

# table files paths
gamma_table_file = os.path.join(tables_path, 'gamma_table.yaml')

xray_table_file = os.path.join(tables_path, 'xray_table.yaml')

element_table_file = os.path.join(tables_path, 'element_table.yaml')

xray_coefficient_table_file = os.path.join(tables_path, 'xray_coefficient_table.yaml')


# load tables
with open(gamma_table_file, 'r') as gt:
    gamma_table = yaml.safe_load(gt)

with open(xray_table_file, 'r') as xt:
    xray_table = yaml.safe_load(xt)

with open(element_table_file, 'r') as et:
    element_table = yaml.safe_load(et)

with open(xray_coefficient_table_file, 'r') as xct:
    xray_coefficient_table = yaml.safe_load(xct)
