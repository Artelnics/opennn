# Benchmarks Page Update Notes

The public Benchmarks page is Elementor-backed. The standard WordPress REST page
object exposes `content.raw`, but not the protected `_elementor_data` custom
field in this installation. The visible card grid is generated from Elementor
layout data, so updating only `content.raw` may not change the live page and may
leave Elementor state inconsistent.

Safe options:

- Use Elementor/admin tooling that exposes and backs up `_elementor_data`.
- Ask a human editor to paste generated card snippets into the Elementor card
grid.
- Add a custom, registered REST/meta endpoint for `_elementor_data` and then
teach the skill to update it.

Do not silently update page `730` with only `content` unless the user explicitly
accepts that limitation.
