import copy
from hashlib import md5
import json
import os
from modules import sd_samplers, shared, scripts, script_callbacks
from modules.script_callbacks import ImageSaveParams
import modules.images as images
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules.shared import opts, OptionInfo
from modules.paths import script_path

import gradio as gr
from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform
from launch import is_installed, run_pip
from modules.generation_parameters_copypaste import parse_generation_parameters
import math
import csv
import random
from logging import PlaceHolder
from dataclasses import dataclass, field
import collections
import copy
from scripts.rl_utils import *


# This is some mega janky code in general that is in bad need of rewriting...
# Please don't look too hard at it.


PARAM_NAMES = ["steps", "cfg_scale", "denoising_strength", "sampler_name"]

extension_name = "Aesthetic Image Scorer"
if platform.system() == "Windows" and not is_installed("pywin32"):
    run_pip(f"install pywin32", "pywin32")
try:
    from tools.add_tags import tag_files
except:
    print(f"{extension_name}: Unable to load Windows tagging script from tools directory")
    tag_files = None

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


try:
    force_cpu = opts.ais_force_cpu
except:
    force_cpu = False

if force_cpu:
    print(f"{extension_name}: Forcing prediction model to run on CPU")
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"

# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device(device=device))

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def remove_black_squares(processed):
    to_remove = [i for i, x in enumerate(processed.images) if round(get_score(x), 1) == 4.6]
    processed.images = [x for i, x in enumerate(processed.images) if i not in to_remove]
    processed.all_prompts = [x for i, x in enumerate(processed.all_prompts) if i not in to_remove]
    processed.infotexts = [x for i, x in enumerate(processed.infotexts) if i not in to_remove]

def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

def get_scores(images):
    scores = []
    for image in images:
        score = get_score(image)
        if round(score, 1) != 4.6:
            scores.append(score)
    return scores


def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features

class AISGroup:
    def __init__(self, name="", apply_choices=lambda choice, choice_values: {"tags": [f"{choice}_{choice_values[choice]}"]}, default=[]):
        gen_params = lambda p: {
            "steps": p.steps,
            "sampler": sd_samplers.samplers[p.sampler_index].name,
            "cfg_scale": p.cfg_scale,
            "seed": p.seed,
            "width": p.width,
            "height": p.height,
            "model": shared.sd_model.sd_model_hash,
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
        }
        self.choice_processors = {
            "aesthetic_score": lambda params: params.pnginfo["aesthetic_score"] if "aesthetic_score" in params.pnginfo else round(get_score(params.image), 1),
            "sampler": lambda params: sd_samplers.samplers[params.p.sampler_index].name if params.p is not None and params.p.sampler else None,
            "cfg_scale": lambda params: params.p.cfg_scale if params.p is not None and params.p.cfg_scale else None,
            "sd_model_hash": lambda params: shared.sd_model.sd_model_hash,
            "seed": lambda params: str(int(params.p.seed)) if params.p is not None and params.p.seed else None,
            "hash": lambda params: md5(json.dumps(gen_params(params.p)).encode('utf-8')).hexdigest() if params.p is not None else None,
        }
        self.name = name
        self.apply_choices = apply_choices
        self.default = default

    def get_choices(self):
        return list(self.choice_processors.keys())

    def get_name(self):
        return self.name

    def selected(self, opts):
        return opts.__getattr__(self.name)

    def get_choice_processors(self):
        return self.choice_processors

    def apply(self, choice_values, applied, opts):
        for choice in self.get_choices():
            if choice in self.selected(opts) and choice in choice_values:
                applied_choices = self.apply_choices(choice, choice_values)
                for key, value in applied_choices.items():
                    if key not in applied:
                        applied[key] = self.get_default()
                    if isinstance(applied[key], dict):
                        applied[key].update(value)
                    else:
                        applied[key] = applied[key] + value
        return applied
    
    def get_default(self):
        return self.default 

class AISGroups:
    def __init__(self, groups=[]):
        self.groups = groups
        self.choices = {}
        self.choice_processors = {}
        for group in groups:
            self.choice_processors.update(group.get_choice_processors())
        self.choices = list(self.choice_processors.keys())

    def all_selected(self, opts):
        selected = {}
        for group in self.groups:
            for select in group.selected(opts):
                selected.update({select: 0})
        return selected.keys()

    def apply(self, opts: list, params: ImageSaveParams):
        parsed_info = parse_generation_parameters(
            params.pnginfo.get("parameters", ""))
        params_hash = md5(json.dumps(
            str(vars(params)), default=lambda o: dir(o), sort_keys=True).encode('utf-8')).hexdigest()
        applied = {}
        if params_hash not in output_cache:
            choice_values = {}
            choices_selected = self.all_selected(opts)
            for choice, processor in self.choice_processors.items():
                if choice in choices_selected:
                    choice_values.update({choice: processor(params)})

            if "seed" in choice_values and choice_values["seed"] is not None and int(choice_values["seed"]) == -1:
                choice_values["seed"] = int(parsed_info["Seed"]) if "Seed" in parsed_info else int(choice_values["seed"])
            
            for group in self.groups:
                applied = group.apply(choice_values, applied, opts)

            expected_keys = ["tags", "categories", "info", "pnginfo"]
            for key in expected_keys:
                if key not in applied:
                    applied[key] = []
            
            output_cache[params_hash] = applied
        else:
            applied = output_cache[params_hash]
            output_cache.clear()
        
        return applied


ais_exif_pnginfo_choices = AISGroup(name="ais_exif_pnginfo_group", apply_choices=lambda choice, choice_values: {
                                              "pnginfo": {choice: choice_values[choice]}}, default={})
ais_windows_tag_group_choices = AISGroup(name="ais_windows_tag_group")
ais_windows_category_group_choices = AISGroup(name="ais_windows_category_group", apply_choices=lambda choice, choice_values: {
                                              "categories": [f"{choice}_{choice_values[choice]}"]})
ais_generation_params_text_choices = AISGroup(name="ais_generation_params_text_group", apply_choices=lambda choice, choice_values: {
                                              "info": {choice: choice_values[choice]}}, default={})

ais_group = AISGroups([
    ais_windows_tag_group_choices,
    ais_windows_category_group_choices,
    ais_generation_params_text_choices,
    ais_exif_pnginfo_choices,
])

output_cache = {}

def on_ui_settings():
    options = {}

    options.update(shared.options_section(('ais', extension_name), {
        ais_exif_pnginfo_choices.get_name(): OptionInfo([], "Save score as EXIF or PNG Info Chunk", gr.CheckboxGroup, {"choices": ais_exif_pnginfo_choices.get_choices()}),
        ais_windows_tag_group_choices.get_name(): OptionInfo([], "Save tags (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_tag_group_choices.get_choices()}),
        ais_windows_category_group_choices.get_name(): OptionInfo([], "Save category (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_category_group_choices.get_choices()}),
        ais_generation_params_text_choices.get_name(): OptionInfo([], "Save generation params text", gr.CheckboxGroup, {"choices": ais_generation_params_text_choices.get_choices()}),
        "ais_force_cpu": OptionInfo(False, "Force CPU (Requires Custom Script Reload)"),
    }))

    opts.add_option(ais_exif_pnginfo_choices.get_name(),
                    options[ais_exif_pnginfo_choices.get_name()])
    opts.add_option(ais_windows_tag_group_choices.get_name(),
                    options[ais_windows_tag_group_choices.get_name()])
    opts.add_option(ais_windows_category_group_choices.get_name(),
                    options[ais_windows_category_group_choices.get_name()])
    opts.add_option(ais_generation_params_text_choices.get_name(),
                    options[ais_generation_params_text_choices.get_name()])
    opts.add_option("ais_force_cpu", options["ais_force_cpu"])

def on_before_image_saved(params: ImageSaveParams):    
    applied = ais_group.apply(opts, params)
    if len(applied["pnginfo"]) > 0:
        params.pnginfo.update(applied["pnginfo"])
    
    if len(applied["info"]) > 0:
        parts = []
        for label, value in applied["info"].items():
            parts.append(f"{label}: {value}")
        if len(parts) > 0:
            if len(params.pnginfo["parameters"]) > 0:
                params.pnginfo["parameters"] += ", "
            params.pnginfo["parameters"] += f"{', '.join(parts)}\n"
    
    return params
        
def on_image_saved(params: ImageSaveParams):
    filename = os.path.realpath(os.path.join(script_path, params.filename))
    applied = ais_group.apply(opts, params)
    if tag_files is not None:
        tag_files(filename=filename, tags=applied["tags"], categories=applied["categories"],
                    log_prefix=f"{extension_name}: ")
    elif platform.system() == "Windows":
        print(f"{extension_name}: Unable to load tagging script")

def add_to_prompt(prompt, weight_dict, offset, seen_list=None, add_to_start = False):
    prompt_list = prompt_to_list(prompt)
    tags, weights = zip(*weight_dict.items())
    seen_list = seen_list + prompt_list
    addables = []
    addables_weights = []
    for i, word in enumerate(tags):
        if word in seen_list:
            seen_list.remove(word)
        else:
            addables.append(word)
            addables_weights.append(weights[i])
    if addables_weights:
        random_tag = weighted_choice(addables, addables_weights, offset)
    else:
        return ()
    prompt = list_to_prompt(prompt_list)
    if add_to_start:
        prompt = random_tag + "," + prompt
    else:
        prompt += "," + random_tag 
    return (prompt, random_tag)


def remove_from_prompt(prompt, weight_dict, offset, seen_list = None):
    prompt_list = prompt_to_list(prompt)
    if seen_list:
        seen_list = seen_list[:]
    tags = list(weight_dict.keys())
    weights = list(weight_dict.values())
    removables = []
    removables_weights = []
    for i, word in enumerate(prompt_list):
        if word in seen_list:
            seen_list.remove(word)
        else:
            removables.append(word)
            removables_weights.append(weights[tags.index(word)])
    if removables_weights:
        random_tag = weighted_choice(removables, removables_weights, offset)
    else:
        return False
    prompt_list.remove(random_tag)
    return (list_to_prompt(prompt_list), random_tag)

@dataclass
class State():
    prompt:                 str = ""
    neg_prompt:             str = ""
    params:                 dict = field(default_factory = lambda: {
                                                                "sampler_name"          : "",
                                                                "steps"                 : 0,
                                                                "cfg_scale"             : 0,
                                                                "denoising_strength"    : 0
                                                                })
    seed:                   int = 0
    visited_prompts:        list = field(default_factory=list) 
    visited_removed:        list = field(default_factory=list) 
    visited_neg_prompts:    list = field(default_factory=list)
    visited_neg_removed:    list = field(default_factory=list) 
    visited_params:         dict = field(default_factory=dict)
    infotexts:              list = field(default_factory=list)
    images:                 list = field(default_factory=list)
    all_prompts:            list = field(default_factory=list)
    score:                  int = 0

    
    def __post_init__(self):
        for key in self.params:
            self.visited_params[key] = [self.params[key]]
    
    def __bool__(self):
        return True

    def __hash__(self):
        return(hash((self.prompt, self.neg_prompt, self.seed, self.params["sampler_name"], self.params["steps"], self.params["cfg_scale"], self.params["denoising_strength"])))
    
    def __eq__(self, other):
        if isinstance(other, State):
            if self.prompt == other.prompt and \
                self.params["steps"] == other.params["steps"] and \
                self.params["cfg_scale"] == other.params["cfg_scale"] and \
                self.params["denoising_strength"] == other.params["denoising_strength"] and \
                self.params["sampler_name"] == other.params["sampler_name"]:
                return True
            else:
                return False
        return False
    
    def copy_p_params_to_state(self, p):
        self.params["sampler_name"] = self.p.sampler_name
        self.params["steps"] = self.p.steps
        self.params["cfg_scale"] = self.p.cfg_scale
        self.params["denoising_strength"] = self.p.denoising_strength
        self.seed = self.p.seed

    
def convert_state_to_p(state, p):
    p.prompt = state.prompt
    p.negative_prompt = state.neg_prompt
    p.steps = state.params["steps"]
    p.cfg_scale = state.params["cfg_scale"]
    p.denoising_strength = state.params["denoising_strength"]
    p.sampler_name = state.params["sampler_name"]
    p.seed = state.seed
    return p
    
class Script(scripts.Script):        
    def title(self):
        return "Score optimiser"    

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        use_file           = gr.Checkbox(label = "Use file called prompts.csv instead of box below", value=False)
        use_file_neg       = gr.Checkbox(label = "Use file called neg_prompts.csv instead of box below", value=False)
        tags_txt           = gr.Textbox(label = "Comma separated prompt, untick the above box if used", lines=1, placeholder="tag1,tag2,tag3(include a tag even if only parameter optimising, it won't be used but prevents bugs)")
        neg_tags_txt       = gr.Textbox(label = "Comma separated negative prompt", lines = 1, placeholder = "tag1,tag2,tag3")
        n_steps            = gr.Number(label = 'Numbers of epochs, total images generated = epochs * batch count', value = 40, precision = 0)
        min_improvement    = gr.Number(label = 'Minimum score improvement to accept tag', value = 0.05)
        n_patience         = gr.Number(label = "Return to initial prompt if stuck for n steps, 0 = no restart", value=10, precision = 0)
        mem_smoothing      = gr.Number(label = "Tag weight offset n > 0, higher = smaller effect of tag weight", value=0.1)
        add_to_start       = gr.Checkbox(label = "Add to start of prompt instead of end", value = False)
        add_chance         = gr.Number(label = "Relative weight of: adding to prompt, 0 = off", value = 3)
        remove_chance      = gr.Number(label = "... removing from prompt", value = 1)
        neg_add_chance     = gr.Number(label = "... adding to neg prompt", value = 0)
        neg_remove_chance  = gr.Number(label = "... removing from neg prompt", value = 0)
        param_chance       = gr.Number(label = "... parameter modification", value = 0)
        seed_change_chance = gr.Number(label = "... seed randomisation", value = 0)
        steps_param        = gr.Textbox(label = "Sampling steps, empty = off for all params", placeholder =  "min, max, step")
        cfg_param          = gr.Textbox(label = "CFG Scale", placeholder = "min, max, step")
        denoise_param      = gr.Textbox(label = "Denoise strength", placeholder = "min, max, step")
        samplers_param     = gr.Textbox(label = "Samplers", placeholder = "some_sampler, another_sampler")
        punc_steps         = gr.Number(label = "Amount of punctuation hacking steps", number = 0)
        return [use_file, use_file_neg, tags_txt, neg_tags_txt, n_steps, min_improvement, n_patience, mem_smoothing, add_to_start, add_chance, remove_chance,
                    neg_add_chance, neg_remove_chance, param_chance, seed_change_chance, steps_param, cfg_param, denoise_param, samplers_param,punc_steps]
    
    def run(self, p, *args):
        params = ["use_file", "use_file_neg", "tags_txt", "neg_tags_txt", "n_steps", "min_improvement", "n_patience", "mem_smoothing", "add_to_start", "add_chance", "remove_chance",
                    "neg_add_chance", "neg_remove_chance", "param_chance", "seed_change_chance", "steps_param", "cfg_param", "denoise_param", "samplers_param","punc_steps"]
        self.__dict__.update(dict(zip(params, args)))
        if p.seed == -1:
            p.seed = random.randint(0, 2**32-1)
        stuck_for = 0
        self.all_params = self.init_params()
        self.all_weights, self.all_improvements = self.init_weights(p)
        first_params = {}
        for param in PARAM_NAMES:
            if param in self.all_params:
                first_params[param] = self.all_params[param][0]
            else:
                exec(f"first_params[param] = p.{param}")
        first_state = State(p.prompt, p.negative_prompt, first_params, p.seed)
        p = convert_state_to_p(first_state, p)
        cur_state = first_state
        best_state = first_state
        all_states = []
        all_scores = []
        for i in range(self.n_steps):
            print(cur_state.visited_params)
            processed = process_images(p)
            remove_black_squares(processed)
            scores = get_scores(processed.images)
            all_scores.extend(scores)
            n_images = len(scores)
            cur_state.score = 1
            if n_images:
                cur_state.score = sum(scores) / n_images
                if i:
                    self.update_weights(cur_state, prev_state, mode, tag)
            self.print_epoch_data(i, cur_state)
            all_states.append(cur_state)
            cur_state.infotexts = [info + f"\nScore:{scores[i]}" for i, info in enumerate(processed.infotexts)]
            cur_state.images = processed.images
            if cur_state.score - self.min_improvement > best_state.score:
                best_state = cur_state
                stuck_for = 0
                print("New best \n")
            else:
                cur_state = best_state
                stuck_for += 1
            if (stuck_for < self.n_patience or not self.n_patience) and (successor := self.get_successor_state(cur_state)):
                print("can_find")
                prev_state = cur_state
                cur_state, mode, tag = successor
                convert_state_to_p(cur_state, p)
            elif self.n_patience and (successor := self.get_successor_state(first_state)):
                stuck_for = 0
                prev_state = first_state
                cur_state, mode, tag = successor
                convert_state_to_p(cur_state, p)
            else:
                break
        self.log_data(all_states, all_scores, p.n_iter*p.batch_size)
        all_states.sort(reverse = True, key = lambda x: x.score)
        overall_best_state = all_states[0]
        print(f"Best score: {overall_best_state.score} \n Prompt: {overall_best_state.prompt} \n Neg_prompt: {overall_best_state.params}")
        images = flatten([state.images for state in all_states])
        all_prompts = flatten(multiply_list([state.prompt for state in all_states], p.n_iter*p.batch_size))
        infotexts = flatten([state.infotexts for state in all_states])
        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)

    def update_weights(self, state, prev_state, mode, tag):
        s_diff = state.score - prev_state.score
        if mode != "seed":
            self.all_weights[mode][tag] += s_diff
            if s_diff > 0:
                self.all_improvements[mode][tag] += s_diff

    def log_data(self,all_states, all_scores, n_iter):
        step_list = multiply_list(list(range(self.n_steps)), n_iter)
        prompt_list = multiply_list([state.prompt for state in all_states], n_iter)
        neg_prompt_list = multiply_list([state.neg_prompt for state in all_states], n_iter)
        cfg_list = multiply_list([state.params["cfg_scale"] for state in all_states], n_iter)
        sampling_steps_list = multiply_list([state.params["steps"] for state in all_states], n_iter)
        denoise_list = multiply_list([state.params["denoising_strength"] for state in all_states], n_iter)
        seed_list = multiply_list([state.seed for state in all_states], n_iter)
        add_list = flatten([list(range(n_iter)) * self.n_steps])
        seed_list = [x + y for x, y in zip(seed_list, add_list)]
        score_list = all_scores
        batch_score_list = multiply_list([state.score for state in all_states],n_iter)
        to_write = list(zip(step_list, prompt_list, neg_prompt_list, cfg_list, sampling_steps_list, denoise_list, seed_list, score_list, batch_score_list))
        write_data_to_csv("./log/optimiser_log", ".csv", ["step", "prompt", "neg_prompt", "cfg", "sampling_steps", "denoise", "seed", "score", "batch_score"], to_write)
        for action, weights in self.all_weights.items():
            to_write = list(zip(weights.keys(), weights.values(), self.all_improvements[action].values()))
            to_write = [(tag, weight, improvement, improvement - weight, improvement / (improvement - weight) if improvement-weight != 0 else 1337) for tag, weight, improvement in to_write]
            write_data_to_csv(f"./log/{action}", ".csv", ["tag", "weight", "total improvement", "difference", "improvement ratio"], to_write)
        
    def init_params(self):
        all_params = {}
        param_vals = [self.steps_param, self.cfg_param, self.denoise_param, self.samplers_param]
        for param, val in zip(PARAM_NAMES, param_vals):
            if val == "":
                continue
            if param == "steps":
                all_params[param] = list(range(*comma_sep_string_to_cast_tuple(val, int)))
            elif param == "sampler_name":
                all_params[param] = comma_sep_to_list(clean_string(val))
            else:
                all_params[param] = frange(*comma_sep_string_to_cast_tuple(val, float))
        return all_params

    def init_weights(self, p):
        all_tags = []
        all_neg_tags = []
        if self.add_chance:
            if self.use_file:
                with open ("prompts.csv", newline="") as f:
                    all_tags = [x for line in list(csv.reader(f)) for x in line]
            else:
                all_tags = self.tags_txt.split(",")
        if self.neg_add_chance:
            if self.use_file_neg:
                with open ("neg_prompts.csv", newline="") as f:
                    all_neg_tags = [x for line in list(csv.reader(f)) for x in line]
            else:
                all_neg_tags = self.neg_tags_txt.split(",")
        all_tags = remove_whitespace_from_list(all_tags)
        all_neg_tags = remove_whitespace_from_list(all_neg_tags)
        prompt_add_weights = {x:0 for x in all_tags}
        prompt_add_improvements = {x:0 for x in all_tags}
        prompt_remove_weights = {x:0 for x in all_tags + prompt_to_list(p.prompt)}
        prompt_remove_improvements = {x:0 for x in all_tags + prompt_to_list(p.prompt)}
        neg_prompt_add_weights = {x:0 for x in all_neg_tags}
        neg_prompt_add_improvements = {x:0 for x in all_neg_tags}
        neg_prompt_remove_weights = {x:0 for x in all_neg_tags + prompt_to_list(p.negative_prompt)}
        neg_prompt_remove_improvements = {x:0 for x in all_neg_tags + prompt_to_list(p.negative_prompt)}
        param_weights = [collections.Counter() for param in PARAM_NAMES]
        param_improvements = [collections.Counter() for param in PARAM_NAMES]
        modes = ["prompt_add", "prompt_remove", "neg_prompt_add", "neg_prompt_remove"] + PARAM_NAMES
        weights = [prompt_add_weights, prompt_remove_weights, neg_prompt_add_weights, neg_prompt_remove_weights] + param_weights
        improvements = [prompt_add_improvements, prompt_remove_improvements, neg_prompt_add_improvements, neg_prompt_remove_improvements] + param_improvements
        all_weights = dict(zip(modes, weights))
        all_improvements = dict(zip(modes, improvements))
        return (all_weights, all_improvements)

    def print_epoch_data(self, i, state):
        print(f"\nStep:{i}/{self.n_steps} \nCurrent score: {state.score} \n Prompt: {state.prompt} \n Negative Prompt: {state.neg_prompt} \n Params: {state.params} \n Seed: {state.seed}\n")

    def get_successor_state(self, state):
        print(state.visited_params)
        if can_do_action := self.find_valid_action(state):
            mode, result, tag = can_do_action
        else:
            return ()
        new_param_dict = copy.deepcopy(state.params)
        new_state = State(state.prompt, state.neg_prompt, new_param_dict, state.seed)
        if mode in PARAM_NAMES:
            state.visited_params[tag] += [result]
            new_state.visited_params[tag] = state.visited_params[tag]
            new_state.params[mode] = result
        if mode == "seed":
            new_state.seed = result
        if mode == "prompt_add":
            state.visited_prompts.append(tag)
            new_state.visited_removed.append(tag)
            new_state.prompt = result
        if mode == "neg_prompt_add":
            state.visited_neg_prompts.append(tag)
            new_state.visited_neg_removed.append(tag)
            new_state.neg_prompt = result
        if mode == "prompt_remove":
            state.prompt_removed.append(tag)
            new_state.prompt = result
        if mode == "neg_prompt_remove":
            state.neg_prompt_removed.append(tag)
            new_state.neg_prompt = result
        return (new_state, mode, tag)

    def find_valid_action(self, state):
        actions = [self.prompt_add, self.prompt_remove, self.param_change, self.neg_prompt_add, self.neg_prompt_remove, self.seed_change]
        actions_chances = [self.add_chance, self.remove_chance, self.param_chance, self.neg_add_chance, self.neg_remove_chance, self.seed_change_chance]
        if not sum(actions_chances):
            actions_chances[0] = 1
        possible_actions = []
        first_action = random.choices(actions, actions_chances)[0]
        possible_actions = [x for i, x in enumerate(actions) if actions_chances[i] != 0 and x != first_action]
        random.shuffle(possible_actions)
        possible_actions = [first_action] + possible_actions
        for action in possible_actions:
            if result := action(state):
                return result
        return ()
    
    def prompt_add(self, state):
        addables = add_to_prompt(state.prompt, self.all_weights["prompt_add"], self.mem_smoothing, state.visited_prompts, self.add_to_start)
        if addables:
            prompt, tag = addables
            state.visited_prompts.append(tag)
            return ("prompt_add", prompt, tag)
        return ()

    def neg_prompt_add(self, state):
        addables = add_to_prompt(state.neg_prompt, self.all_weights["neg_prompt_add"], self.mem_smoothing, state.visited_neg_prompts, self.add_to_start)
        if addables:
            neg_prompt, tag = addables
            state.visited_neg_prompts.append(tag)
            return ("neg_prompt_add", neg_prompt, tag)
        return ()

    def prompt_remove(self, state):
        if prompt_length(state.prompt) > 2 and len(state.visited_removed) < prompt_length(state.prompt):
            removables = remove_from_prompt(state.prompt, self.all_weights["prompt_remove"], self.mem_smoothing, state.visited_removed)
            if removables:
                prompt, tag = removables
                state.visited_removed.append(tag)
                return ("prompt_remove", prompt, tag)
        return ()

    def neg_prompt_remove(self, state):
        if prompt_length(state.neg_prompt) > 2 and len(state.visited_neg_removed) < prompt_length(state.neg_prompt):
            removables = remove_from_prompt(state.neg_prompt, self.all_weights["neg_prompt_remove"], self.mem_smoothing, state.visited_neg_removed)
            if removables:
                neg_prompt, tag = removables
                state.visited_neg_removed.append(tag)
                return ("neg_prompt_remove", neg_prompt, tag)
        return ()

    def param_change(self, state):
        shuffled_params = copy.deepcopy(list(self.all_params.items()))
        n_params = len(shuffled_params)
        random.shuffle(shuffled_params)
        for i in range(n_params):
            param, possible_param_vals = shuffled_params.pop()
            if len(possible_param_vals) == len(state.visited_params[param]):
                continue
            print(possible_param_vals, state.visited_params[param])
            new_param_val = pick_unique(possible_param_vals, state.visited_params[param])
            return (param, new_param_val, param)
        return ()

    def seed_change(self, state):
        new_seed = random.randint(0, 2**32-1)
        return ("seed", new_seed, new_seed)

    def punc_hack(self):
        pass

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_before_image_saved)
script_callbacks.on_image_saved(on_image_saved)