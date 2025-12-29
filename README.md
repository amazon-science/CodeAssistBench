# Project Title

This project aims to accommodate various use cases by determining the best approach for each scenario. The system reads configuration from multiple sources.

## Features
- Easy to use API
- Flexible configuration options
- Works across multiple platforms

## Installation
Follow the instructions to install the package.

## Usage
See documentation for usage examples.

Dropping a link to a Figma file in the composer and asking Cursor to do something with it should automatically trigger the get_figma_data tool.

## Available Tools
The server provides the following MCP tools:

### get_figma_data
Fetches simplified layout and styling data from a Figma file. If nodeId is provided, it fetches only that node (and optionally its subtree). If nodeId is omitted, it fetches the entire file.

Parameters:
- fileKey (string): The key of the Figma file to fetch, often found in a URL like figma.com/(file|design)/<fileKey>/...
- nodeId (string, optional): The node to fetch, usually from the URL parameter node-id=<nodeId>. Use when available.
- depth (number, optional): How many levels deep to traverse the node tree. Only use when explicitly needed.

Response:
- Returns a text payload containing JSON with shape:
  { "metadata": { ... }, "nodes": [ ... ], "globalVars": { ... } }

Example:
```json
{
  "fileKey": "AbCdEfGhIjKlMnOpQrStUv",
  "nodeId": "1234:5678",
  "depth": 2
}
```

### download_figma_images
Downloads SVG and PNG images referenced in a Figma file for the provided nodes. For nodes with image fills, include imageRef. For vector nodes, omit imageRef and select the desired format by the fileName extension (.svg for SVG, anything else results in PNG).

Parameters:
- fileKey (string): The key of the Figma file containing the nodes
- nodes (array): Each item contains:
  - nodeId (string): The Figma node ID, formatted like 1234:5678
  - imageRef (string, optional): Required for fill-based images; omit for vector nodes
  - fileName (string): Local file name to save as; .svg triggers SVG export, otherwise PNG
- localPath (string): Absolute path to the directory where images are saved. Directories are created as needed.

Example:
```json
{
  "fileKey": "AbCdEfGhIjKlMnOpQrStUv",
  "nodes": [
    { "nodeId": "101:200", "fileName": "icons/close.svg" },
    { "nodeId": "101:201", "imageRef": "fbdcdf45", "fileName": "images/hero.png" }
  ],
  "localPath": "/absolute/path/to/public/assets"
}
```

### Note on tool changes
- Replaced tools: The previous get_file and get_node tools have been consolidated into get_figma_data.
  - get_file → use get_figma_data with fileKey (and optional depth)
  - get_node → use get_figma_data with fileKey and nodeId (and optional depth)
- Functional differences:
  - Consolidation simplifies agent prompting and allows a single entry point whether you have a node link or a full file link.
  - The response is returned as a JSON string with metadata, nodes, and globalVars, optimized for LLM consumption and context limits.
- New capability: download_figma_images enables fetching actual assets (SVG/PNG) and saving them locally, which was not supported by the older tools.
