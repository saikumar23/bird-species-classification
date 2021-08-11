from __future__ import division, print_function
import torch
import torchvision.transforms as tt
from PIL import Image
import torch.nn as nn
from flask import Flask, redirect, url_for, request, render_template
from torchvision.transforms import transforms
from werkzeug.utils import secure_filename
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
#from gevent.pywsgi import WSGIServer
app = Flask(__name__)
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Input: 64 x 3 x 64 x 64
        self.conv1 = conv_block(in_channels, 64) # 64 x 64 x 64 x 64
        self.conv2 = conv_block(64, 128, pool=True) # 64 x 128 x 32 x 32
        self.res1 = nn.Sequential(conv_block(128, 128), # 64 x 128 x 32 x 32
                                  conv_block(128, 128)) # 64 x 128 x 32 x 32
        
        self.conv3 = conv_block(128, 256, pool=True) # 64 x 256 x 16 x 16
        self.conv4 = conv_block(256, 512, pool=True) # 64 x 512 x 8 x 8 
        self.res2 = nn.Sequential(conv_block(512, 512), # 64 x 512 x 8 x 8 
                                  conv_block(512, 512)) # 64 x 512 x 8 x 8 
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), # 64 x 512 x 1 x 1 
                                        nn.Flatten(), # 64 x 512
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
model = torch.load('cnn.pt',map_location='cpu')
print('Model loaded')
tts = tt.Compose([tt.Resize(64),tt.RandomCrop(64),tt.ToTensor()])
l = ['AFRICAN CROWNED CRANE',
 'AFRICAN FIREFINCH',
 'ALBATROSS',
 'ALEXANDRINE PARAKEET',
 'AMERICAN AVOCET',
 'AMERICAN BITTERN',
 'AMERICAN COOT',
 'AMERICAN GOLDFINCH',
 'AMERICAN KESTREL',
 'AMERICAN PIPIT',
 'AMERICAN REDSTART',
 'ANHINGA',
 'ANNAS HUMMINGBIRD',
 'ANTBIRD',
 'ARARIPE MANAKIN',
 'ASIAN CRESTED IBIS',
 'BALD EAGLE',
 'BALI STARLING',
 'BALTIMORE ORIOLE',
 'BANANAQUIT',
 'BANDED BROADBILL',
 'BAR-TAILED GODWIT',
 'BARN OWL',
 'BARN SWALLOW',
 'BARRED PUFFBIRD',
 'BAY-BREASTED WARBLER',
 'BEARDED BARBET',
 'BEARDED REEDLING',
 'BELTED KINGFISHER',
 'BIRD OF PARADISE',
 'BLACK & YELLOW bROADBILL',
 'BLACK FRANCOLIN',
 'BLACK SKIMMER',
 'BLACK SWAN',
 'BLACK TAIL CRAKE',
 'BLACK THROATED BUSHTIT',
 'BLACK THROATED WARBLER',
 'BLACK VULTURE',
 'BLACK-CAPPED CHICKADEE',
 'BLACK-NECKED GREBE',
 'BLACK-THROATED SPARROW',
 'BLACKBURNIAM WARBLER',
 'BLUE GROUSE',
 'BLUE HERON',
 'BOBOLINK',
 'BORNEAN BRISTLEHEAD',
 'BORNEAN LEAFBIRD',
 'BROWN NOODY',
 'BROWN THRASHER',
 'BULWERS PHEASANT',
 'CACTUS WREN',
 'CALIFORNIA CONDOR',
 'CALIFORNIA GULL',
 'CALIFORNIA QUAIL',
 'CANARY',
 'CAPE MAY WARBLER',
 'CAPUCHINBIRD',
 'CARMINE BEE-EATER',
 'CASPIAN TERN',
 'CASSOWARY',
 'CEDAR WAXWING',
 'CHARA DE COLLAR',
 'CHIPPING SPARROW',
 'CHUKAR PARTRIDGE',
 'CINNAMON TEAL',
 'CLARKS NUTCRACKER',
 'COCK OF THE  ROCK',
 'COCKATOO',
 'COMMON FIRECREST',
 'COMMON GRACKLE',
 'COMMON HOUSE MARTIN',
 'COMMON LOON',
 'COMMON POORWILL',
 'COMMON STARLING',
 'COUCHS KINGBIRD',
 'CRESTED AUKLET',
 'CRESTED CARACARA',
 'CRESTED NUTHATCH',
 'CROW',
 'CROWNED PIGEON',
 'CUBAN TODY',
 'CURL CRESTED ARACURI',
 'D-ARNAUDS BARBET',
 'DARK EYED JUNCO',
 'DOUBLE BARRED FINCH',
 'DOWNY WOODPECKER',
 'EASTERN BLUEBIRD',
 'EASTERN MEADOWLARK',
 'EASTERN ROSELLA',
 'EASTERN TOWEE',
 'ELEGANT TROGON',
 'ELLIOTS  PHEASANT',
 'EMPEROR PENGUIN',
 'EMU',
 'ENGGANO MYNA',
 'EURASIAN GOLDEN ORIOLE',
 'EURASIAN MAGPIE',
 'EVENING GROSBEAK',
 'FIRE TAILLED MYZORNIS',
 'FLAME TANAGER',
 'FLAMINGO',
 'FRIGATE',
 'GAMBELS QUAIL',
 'GANG GANG COCKATOO',
 'GILA WOODPECKER',
 'GILDED FLICKER',
 'GLOSSY IBIS',
 'GO AWAY BIRD',
 'GOLD WING WARBLER',
 'GOLDEN CHEEKED WARBLER',
 'GOLDEN CHLOROPHONIA',
 'GOLDEN EAGLE',
 'GOLDEN PHEASANT',
 'GOLDEN PIPIT',
 'GOULDIAN FINCH',
 'GRAY CATBIRD',
 'GRAY PARTRIDGE',
 'GREAT POTOO',
 'GREATOR SAGE GROUSE',
 'GREEN JAY',
 'GREEN MAGPIE',
 'GREY PLOVER',
 'GUINEA TURACO',
 'GUINEAFOWL',
 'GYRFALCON',
 'HARPY EAGLE',
 'HAWAIIAN GOOSE',
 'HELMET VANGA',
 'HIMALAYAN MONAL',
 'HOATZIN',
 'HOODED MERGANSER',
 'HOOPOES',
 'HORNBILL',
 'HORNED GUAN',
 'HORNED SUNGEM',
 'HOUSE FINCH',
 'HOUSE SPARROW',
 'IMPERIAL SHAQ',
 'INCA TERN',
 'INDIAN BUSTARD',
 'INDIAN PITTA',
 'INDIGO BUNTING',
 'JABIRU',
 'JAVA SPARROW',
 'KAKAPO',
 'KILLDEAR',
 'KING VULTURE',
 'KIWI',
 'KOOKABURRA',
 'LARK BUNTING',
 'LEARS MACAW',
 'LILAC ROLLER',
 'LONG-EARED OWL',
 'MAGPIE GOOSE',
 'MALABAR HORNBILL',
 'MALACHITE KINGFISHER',
 'MALEO',
 'MALLARD DUCK',
 'MANDRIN DUCK',
 'MARABOU STORK',
 'MASKED BOOBY',
 'MASKED LAPWING',
 'MIKADO  PHEASANT',
 'MOURNING DOVE',
 'MYNA',
 'NICOBAR PIGEON',
 'NOISY FRIARBIRD',
 'NORTHERN BALD IBIS',
 'NORTHERN CARDINAL',
 'NORTHERN FLICKER',
 'NORTHERN GANNET',
 'NORTHERN GOSHAWK',
 'NORTHERN JACANA',
 'NORTHERN MOCKINGBIRD',
 'NORTHERN PARULA',
 'NORTHERN RED BISHOP',
 'NORTHERN SHOVELER',
 'OCELLATED TURKEY',
 'OKINAWA RAIL',
 'OSPREY',
 'OSTRICH',
 'OVENBIRD',
 'OYSTER CATCHER',
 'PAINTED BUNTIG',
 'PALILA',
 'PARADISE TANAGER',
 'PARAKETT  AKULET',
 'PARUS MAJOR',
 'PEACOCK',
 'PELICAN',
 'PEREGRINE FALCON',
 'PHILIPPINE EAGLE',
 'PINK ROBIN',
 'PUFFIN',
 'PURPLE FINCH',
 'PURPLE GALLINULE',
 'PURPLE MARTIN',
 'PURPLE SWAMPHEN',
 'PYGMY KINGFISHER',
 'QUETZAL',
 'RAINBOW LORIKEET',
 'RAZORBILL',
 'RED BEARDED BEE EATER',
 'RED BELLIED PITTA',
 'RED BROWED FINCH',
 'RED FACED CORMORANT',
 'RED FACED WARBLER',
 'RED HEADED DUCK',
 'RED HEADED WOODPECKER',
 'RED HONEY CREEPER',
 'RED TAILED THRUSH',
 'RED WINGED BLACKBIRD',
 'RED WISKERED BULBUL',
 'REGENT BOWERBIRD',
 'RING-NECKED PHEASANT',
 'ROADRUNNER',
 'ROBIN',
 'ROCK DOVE',
 'ROSY FACED LOVEBIRD',
 'ROUGH LEG BUZZARD',
 'ROYAL FLYCATCHER',
 'RUBY THROATED HUMMINGBIRD',
 'RUFOUS KINGFISHER',
 'RUFUOS MOTMOT',
 'SAMATRAN THRUSH',
 'SAND MARTIN',
 'SCARLET IBIS',
 'SCARLET MACAW',
 'SHOEBILL',
 'SHORT BILLED DOWITCHER',
 'SMITHS LONGSPUR',
 'SNOWY EGRET',
 'SNOWY OWL',
 'SORA',
 'SPANGLED COTINGA',
 'SPLENDID WREN',
 'SPOON BILED SANDPIPER',
 'SPOONBILL',
 'SRI LANKA BLUE MAGPIE',
 'STEAMER DUCK',
 'STORK BILLED KINGFISHER',
 'STRAWBERRY FINCH',
 'STRIPPED SWALLOW',
 'SUPERB STARLING',
 'SWINHOES PHEASANT',
 'TAIWAN MAGPIE',
 'TAKAHE',
 'TASMANIAN HEN',
 'TEAL DUCK',
 'TIT MOUSE',
 'TOUCHAN',
 'TOWNSENDS WARBLER',
 'TREE SWALLOW',
 'TRUMPTER SWAN',
 'TURKEY VULTURE',
 'TURQUOISE MOTMOT',
 'UMBRELLA BIRD',
 'VARIED THRUSH',
 'VENEZUELIAN TROUPIAL',
 'VERMILION FLYCATHER',
 'VICTORIA CROWNED PIGEON',
 'VIOLET GREEN SWALLOW',
 'VULTURINE GUINEAFOWL',
 'WATTLED CURASSOW',
 'WHIMBREL',
 'WHITE CHEEKED TURACO',
 'WHITE NECKED RAVEN',
 'WHITE TAILED TROPIC',
 'WHITE THROATED BEE EATER',
 'WILD TURKEY',
 'WILSONS BIRD OF PARADISE',
 'WOOD DUCK',
 'YELLOW BELLIED FLOWERPECKER',
 'YELLOW CACIQUE',
 'YELLOW HEADED BLACKBIRD']
def model_predict(img_path,model,tts,l):
    img = Image.open(img_path)
    trans_img = tts(img)
    trans_img = trans_img.unsqueeze(0)
    s = model(trans_img)
    _, preds  = torch.max(s, dim=1)
    return l[preds[0].item()]
print('predict model ran')
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model,tts,l)
        result=preds
        return result
    return None
print('Everything ran')
if __name__ == '__main__':
    app.run(debug=True)

