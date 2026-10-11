# ANSI images

Images are not well-supported by the [Textual](https://textual.textualize.io)
library. It is easier to convert them to raw ANSI codes using a third-party
program, and to display them using Rich's `Text.from_ansi`.

The ansi files in this directory are made using
[chafa](https://hpjansson.org/chafa/), and fiddling around with the settings eg:

```bash

chafa -s 40x40 logo.png > splash.ansi
chafa  -s 20x20 --color-extractor median -p true logo.png > logo.ansi

```
