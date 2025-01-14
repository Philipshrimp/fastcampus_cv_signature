import roboflow


roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace("team-roboflow").project("coco-128")
dataset = project.version(2).download("coco")
