# -*- coding: utf-8 -*-

from __future__ import print_function
import json

from evaluate import infer
from DB import Database

from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat
import os
from shutil import copyfile

depth = 5
d_type = 'd1'
query_idx = 15



if __name__ == '__main__':
  db = Database()

  # retrieve by color
  method = Color()
  samples = method.make_samples(db)
  query = samples[query_idx]
  print(samples)
  print("QUERY: ", query, len(query))
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by daisy
  method = Daisy()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by edge
  method = Edge()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by gabor
  method = Gabor()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by HOG
  method = HOG()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by VGG
  method = VGGNetFeat()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by resnet
  method = ResNetFeat()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)
  
  #saving path to output_results.txt

  output_file_path = 'output_results.txt' 

  with open(output_file_path, 'w') as output_file:
        for r in result:
            output_file.write(r['img'] + '\n')

  print(f"Results saved to {output_file_path}")




  # Filter lines based on the domain
  domain="Clipart"

  input_file_path = 'output_results.txt'
  with open(input_file_path, 'r') as input_file:
    lines = input_file.read().splitlines()

  art_lines = [line for line in lines if domain in line]

  sorted_art_lines = sorted(art_lines)

  output_file_path = 'sorted_domain_results.txt'
  with open(output_file_path, 'w') as output_file:
    for line in sorted_art_lines:
        output_file.write(line + '\n')

  print(f"Sorted {domain} results saved to {output_file_path}")


  #image copy to output_sorted_images folder
  with open('sorted_domain_results.txt', 'r') as file:
    image_paths = file.read().splitlines()

  output_sorted_folder = 'output_sorted_images'

  os.makedirs(output_sorted_folder, exist_ok=True)

  for path in image_paths:
    source_path = os.path.join('.', path) 
    destination_path = os.path.join(output_sorted_folder, os.path.basename(path))

    copyfile(source_path, destination_path)

  print(f"Images copied to {output_sorted_folder}")