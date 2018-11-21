#!/usr/bin/env python3

from PIL import Image
import discord
import json
import os
import math
import pickle
import random

# Constants
AUTH_FILE = 'auth.json'
AUTH_FIELD = 'token'

MAX_EMOJI_ROW = 50
MAX_EMOJI_COL = 23
EMOJI_DIM = 32

MAX_IMG_HEIGHT = MAX_EMOJI_COL * EMOJI_DIM
MAX_IMG_WIDTH = MAX_EMOJI_ROW * EMOJI_DIM

EMOJI_DATA_FILE = 'emoji-data.txt'
EMOJI_CACHE_FILE = 'emoji_cache.txt'
EMOJI_RES_DIR = 'res/'

MAX_PRINTABLE_ROWS = 5

PICKLE_FILE = 'emoji_tree.pickle'

COMMAND_PREFIX = '?'

BOT_OWNER = 141102297550684160

# K-D tree for fast NN lookup
class KDNode:
    def __init__(self, key, val, left=None, right=None):
        '''
            axis: axis of the node
            key: has at least [axis] dimensions
            val: value of node
            left: left subtree
            right: right subtree
        '''
        self._key = key
        self.val = val
        self._left = left
        self._right = right

    def insert(self, key, val, depth=0):
        axis = depth % len(key)
        if key[axis] < self._key[axis]:
            if self._left is None:
                self._left = KDNode(key, val)
            else:
                self._left.insert(key, val, depth + 1)
        else:
            if self._right is None:
                self._right = KDNode(key, val)
            else:
                self._right.insert(key, val, depth + 1)

    def nearest_emoji(self, key):
        return self.nearest_neighbor(key)[1]

    def nearest_neighbor(self, key, depth=0):
        best = (self._key, self.val)
        # At leaf node
        if self._left is None and self._right is None:
            return best

        best_dist = euclid_squared(self._key, key)
        #print('Key: {}; Depth: {}; Dist: {}'.format(self._key, depth, math.sqrt(best_dist)))
        axis = depth % len(key)

        # Find best in subtree
        best_subtree = None
        hypersphere_check = None
        if key[axis] < self._key[axis]:
            if self._left is not None:
                best_subtree = self._left.nearest_neighbor(key, depth + 1)
            hypersphere_check = self._right
        if key[axis] > self._key[axis]:
            if self._right is not None:
                best_subtree = self._right.nearest_neighbor(key, depth + 1)
            hypersphere_check = self._left

        if best_subtree is not None:
            # Check if subtree has better result
            best_subtree_dist = euclid_squared(best_subtree[0], key)
            if best_subtree_dist < best_dist:
                best = best_subtree
                best_dist = best_subtree_dist

        # Check if hyperplane intersects hypersphere
        if hypersphere_check is not None:
            radius = euclid(key, best[0])
            if abs(key[axis] - hypersphere_check._key[axis]) <= radius:
                hypersphere_best = hypersphere_check.nearest_neighbor(key, depth + 1)
                hypersphere_dist = euclid_squared(hypersphere_best[0], key)
                if hypersphere_dist < best_dist:
                    best = hypersphere_best

        return best

def euclid(p1, p2):
    '''
        Returns the euclidian distance between p1 and p2
    '''
    return math.sqrt(euclid_squared(p1, p2))


def euclid_squared(p1, p2):
    '''
        Returns the euclidian distance squared between p1 and p2
    '''
    return sum([(x[0] - x[1]) * (x[0] - x[1]) for x in zip(p1, p2)])

def construct_kdtree(data, depth=0):
    '''
        data: list of items to insert into tree
        depth: tree depth
    '''
    if not data:
        return None

    # Axis to split on
    axis = depth % len(data[0])

    # Sort and get median
    data.sort(key=lambda x: x[0][axis])
    median = len(data) // 2

    # Construct node and split data
    return KDNode(data[median][0],
                  data[median][1],
                  construct_kdtree(data[:median], depth + 1),
                  construct_kdtree(data[median + 1:], depth + 1))

# discord client
client = discord.Client()

# KD Tree storing emoji with color value
emoji_tree = None

def __main__():
    client.run(retrieve_token())

# Events

@client.event
async def on_ready():
    global emoji_tree
    if os.path.isfile(PICKLE_FILE):
        print('Cached file found, loading into memory')
        with open(PICKLE_FILE, 'rb') as f:
            emoji_tree = pickle.load(f)
    else:
        print('Indexing emojis')
        index_emojis()
    print('Bot is ready')

@client.event
async def on_message(message):
    if not message.author.bot:
        if message.content.startswith(COMMAND_PREFIX):
            command = message.content[1:]
            if command.startswith('help'):
                print('Help invoked')
                await display_help(message)
            elif command.startswith('trout'):
                print('Trout invoked')
                await trout(message)
            elif command.startswith('dildo'):
                print('Trout invoked')
                await trout(message, 'dildo')
            elif command.startswith('roll'):
                print('Roll invoked')
                await roll(message)
            elif command.startswith('mosaicify'):
                print('Mosaicify invoked')
                await mosaicify(message.channel, message.attachments)

async def mosaicify(channel, image):
    if not image or image[0].height is None:
        await channel.send('Error: Expected an image attachment')
    else:
        image = image[0]
        written = await image.save(open(image.filename, 'wb'))
        if written != image.size:
            await channel.send('Unexpected error processing attachment, please try again.')
        else:
            mosaic = compute_mosaic(image.filename)
            if mosaic is not None:
                for row in mosaic:
                    await channel.send(row)
            else:
                await channel.send('Error: Image is not supported or has bad extension type')
            os.remove(image.filename)

async def roll(message):
    channel = message.channel
    
    data = message.content.split()

    if 1 <= len(data) <= 2:
        num_dices, faces = data[1].split('d') if len(data) > 1 else ['1', '6']
        num_dices = int(num_dices) if num_dices else 1
        faces = int(faces)

        if num_dices > 20:
            await channel.send('You\'re getting a bit ambitious with the number of die')
        elif faces > 0:
            rolls = [random.randint(1, faces) for d in range(num_dices)]
            if num_dices > 1:
                await channel.send('Roll: {} = {}'.format(' + '.join(map(str, rolls)), sum(rolls)))
            else:
                await channel.send('Rolled a {}'.format(rolls[0]))
        else:
            await channel.send('Invalid number of faces on die')
    else:
        await channel.send('Error, invalid format. Invoke \'{}help roll\' for command usage'
                           .format(COMMAND_PREFIX))

def compute_mosaic(filename):
    emoji = []
    img = resize_bounds(Image.open(filename).convert('RGB'))
    print('Image dim is {}x{}'.format(img.width, img.height))
    emoji_str = ''
    row_count = 0
    for y in range(0, img.height, EMOJI_DIM):
        for x in range(0, img.width, EMOJI_DIM):
            avg_color = calc_avg_color(img.crop((x, y, x + EMOJI_DIM, y + EMOJI_DIM)))
            emoji_str += emoji_tree.nearest_emoji(avg_color)
        row_count += 1
        if row_count >= MAX_PRINTABLE_ROWS:
            emoji.append(emoji_str)
            emoji_str = ''
            row_count = 0
        else:
            emoji_str += '\n'

    print('Computed emoji\n{}'.format('\n'.join(emoji)))
    return emoji


def resize_bounds(img):
    '''
        Resizes given image and constrains it within max dimensions.
    '''
    ratio = 1
    if img.width > MAX_IMG_WIDTH or img.height > MAX_IMG_HEIGHT:
        if img.width > img.height:
            ratio = img.width / MAX_IMG_WIDTH
        else:
            ratio = img.height / MAX_IMG_HEIGHT
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


async def display_help(message):
    channel = message.channel

    data = message.content.split()

    if len(data) == 1:
        await channel.send('''\
?help - displays this help message
?trout <user> - trouts trouts the user
?roll <xdy> - rolls some dice
?mosaicify <image> - converts image into mosaic
''')
    elif len(data) == 2:
        command = data[1]
        if command == 'help':
            await channel.send('Are you joking right now?')
        elif command == 'trout':
            await channel.send('Used splash but it was innefective')
        elif command == 'roll':
            await channel.send('xdy - roll x die with y faces')
        elif command == 'mosaicify':
            await channel.send('Upload an image with the description invoking this command')
        else:
            await channel.send('Could not find detailed help for given command')

async def trout(message, weapon='trout'):
    '''
        Slap the recipient with the specified weapon. Can't target bot's owner.
    '''
    channel = message.channel
    author = message.author
    target = message.mentions

    if len(target) != 1:
        await channel.send('Slaps {} in the face with a {} for incorrect command usage'
                           .format(author.mention, weapon))
    else:
        target = target[0]
        if target.id == BOT_OWNER or target == client.user:
            await channel.send('Shoves a {} up {}\'s ass for disobediance'
                               .format(weapon, author.mention))
        else:
            await message.delete()
            await channel.send('Slaps {} in the face with a {}'.format(target.mention, weapon))

# Index available emojis for use
def index_emojis():
    global emoji_tree
    emoji_list = []

    with open(EMOJI_DATA_FILE, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith('#'):
                # HEX ; <information>
                # HEX..HEX ; <information>
                codepoint = line.split(';')[0].strip()
                if '..' in codepoint:
                    start, end = codepoint.split('..')
                else:
                    start = end = codepoint

                for codepoint_val in range(int(start, 16), int(end, 16) + 1):
                    codepoint_str = hex(codepoint_val)[2:]
                    emoji_file = EMOJI_RES_DIR + codepoint_str + '.png'
                    if os.path.isfile(emoji_file):
                        img = Image.open(emoji_file)
                        avg_color = calc_avg_color(img.convert('RGB'))
                        print('File {} has average color {}'.format(emoji_file, avg_color))
                        emoji_list.append((avg_color, chr(int(codepoint_str, 16))))

    emoji_tree = construct_kdtree(emoji_list)
    
    # Serialize data for future loads
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(emoji_tree, f)

def calc_avg_color(img):
    total_color = tuple(sum(color) for color in zip(*img.getdata()))
    return tuple(map(lambda x: x // (img.width * img.height), total_color))

# Load token to run bot
def retrieve_token():
    return json.load(open(AUTH_FILE, 'r'))[AUTH_FIELD]

if __name__ == '__main__':
    __main__()
