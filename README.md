## Custom nodes for ComfyUI

Currently includes simple nodes that I wanted for easier and more compact workflow creation, and flow control nodes.

~~The Flow control nodes require my custom execution.py at [comfyui-flow-control](https://github.com/SirVeggie/comfyui-flow-control).~~
After the execution model inversion, my custom execution model is no longer compatible with latest comfy, and the relevant nodes have been removed at least for now. Instead new nodes that are compatible with the execution model inversion have now been added. I will probably try to add caching nodes back at some point.

Beware of bugs.

### Flow control

Note: the custom execution.py file may get out of date quickly if the execution file is updated by comfy.

The flow control nodes are powerful nodes that allow skipping nodes in the middle of the flow without muting or bypassing. This allows for automating flow control without wasting processing, since nodes can be skipped completely preventing their execution. The flow can be controlled on-the-fly during generation based on calculated conditions.

Example: Your flow can take a completely different branch depending on what objects were detected in an image, or how many faces an adetailer node found.

### Documentation

See the [wiki](https://github.com/SirVeggie/comfyui-sv-nodes/wiki).

## Node list

This list is out of date. :)

### Vanilla nodes

- Prompt Processing: see [style-vars](https://github.com/SirVeggie/extension-style-vars) extension for automatic1111
- Resolution Selector
- Resolution Selector 2
- Basic Parameters: combining some common settings for ksamplers
- String Separator: split text into two parts (useful for positive and negative prompt)
- Check None: true if input was None, useful for flow control
- Any to Any
- My version of `Pipes`
- etc.

### Flow nodes

- Prompt + Model: requires the caching improvements of flow control
- Cache Shield: prevent a node from re-executing if inputs remain the same
- Manual Cache: can keep an output in cache even if it changes later
- Flow Block: prevent all subsequent nodes from executing, the input nodes are also not executed if not needed elsewhere
- Flow Block Signal: combine with flow block
- Flow Block Simple: combines the previous two nodes, does not prevent input nodes from executing unless the toggle is a primitive
- Flow Continue: unblocks a blocked flow further down the flow, and uses the first non-blocked input, None if all inputs are blocked
- Flow Continue Simple: unblocks a blocked flow, outputs None if flow was blocked
- Flow Node: simple many-to-many node, useful for blocking many things at once, block first input to prevent the following inputs' upstream nodes from executing
