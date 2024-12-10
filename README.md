# WIP

Steps to run this codebase:

0. Make sure you have [Pandoc](https://pandoc.org/) installed and its CLI added to the PATH
1. `pip install -r requirements.txt` (installs dependencies)
2. run `python convert_to_excel.py` to generate a `projects.xlsx` or bring your own xlsx sheet to process later on
3. generate articles using `python gen_article.py --projects-file your/sheet.xlsx` (`--projects-file ...` is optional, by default it looks for `resources/projects.xlsx`)
4. upload articles to wiki using `python upload_articles.py`

NOTE: Ensure you have a `.env` file properly set up based on the `.env.example` file before running the scripts.

## License

This project is licensed under the terms of the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

![CC BY 4.0 License](https://i.creativecommons.org/l/by/4.0/88x31.png)
