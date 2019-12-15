import unittest
import ImgSplit
import ImgSplit_refactor
import os
import random

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

class Test_ImgSplit_test(unittest.TestCase):
    def setUp(self):
        expect = ImgSplit.splitbase(r'/data/dj/dota/val',
                       r'/data/dj/dota/val_1024_debugmulti_expect')
        expect.splitdata(0.4)

        refactor = ImgSplit.splitbase(r'/data/dj/dota/val',
                                      r'/data/dj/dota/val_1024_debugmulti_refactor')
        refactor.splitdata(0.4)

    def test_output_names(self):

        expect_filenames = GetFileFromThisRootDir(os.path.join(r'/data/dj/dota/val_1024_debugmulti_expect', 'images'))

        refactor_filenames = GetFileFromThisRootDir(os.path.join(r'/data/dj/dota/val_1024_debugmulti_refactor/images'))

        for index in range(len(expect_filenames)):
            self.assertEqual(expect_filenames[index], refactor_filenames[index])



        nums = [random.randint(0, len(expect_filenames)) for _ in range(30)]
        #
        # for num in nums:
        #     exect_name =


