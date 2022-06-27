import os
import unittest
import tempfile
from datetime import datetime

from gazettes.preprocessing import (
    remove_word_with_duplicate_letters,
    remove_duplicate_whitespaces,
    remove_consecutive_empty_lines,
    find_duplicates_chars,
    remove_new_line_char,
    remove_line_with_punctuation_only,
    preprocess_gazette_txt_file,
    sentence_segmentation,
    create_sentence_file,
    find_gazette_files,
    _is_directory_date,
    remove_special_line_prefix,
    remove_special_quotes,
)


class PreprocessingTests(unittest.TestCase):
    def test_remove_consecutive_empty_lines(self):
        test_cases = [
            ("\n", "\n"),
            ("\n\n", "\n"),
            ("\n\n\n", "\n"),
            ("\n\n\n\n", "\n"),
            ("\n\n\n\n\n", "\n"),
        ]
        for inputt, expected_output in test_cases:
            output = remove_consecutive_empty_lines(inputt)
            self.assertEqual(output, expected_output)

    def test_remove_empty_lines(self):
        test_cases = [
            ("\n", ""),
            ("\n\n", ""),
            ("\n\n\n", ""),
            ("\n\n\n\n", ""),
            ("\n\n\n\n\n", ""),
            ("ab\ncd", "ab cd"),
            ("ab\n\ncd", "ab cd"),
            ("ab\n\n\ncd", "ab cd"),
            ("ab\n\n\n\ncd", "ab cd"),
            ("ab\n\n\n\ncd\nef\n\n", "ab cd ef"),
        ]
        for inputt, expected_output in test_cases:
            output = remove_new_line_char(inputt)
            self.assertEqual(output, expected_output)

    def test_remove_duplicate_whitespaces(self):
        test_cases = [
            (" Oi    meu  nome     e fulano  ", " Oi meu nome e fulano "),
            ("    ", " "),
            ("   ", " "),
            ("  ", " "),
            ("    ", " "),
            (
                "  Oi    meu  nome     e fulano  \n  E o meu eh ciclano  ",
                " Oi meu nome e fulano \n E o meu eh ciclano ",
            ),
        ]
        for inputt, expected_output in test_cases:
            output = remove_duplicate_whitespaces(inputt)
            self.assertEqual(output, expected_output)

    def test_find_duplicate_char(self):
        output = find_duplicates_chars("aabbcc dd ee ffgghh")
        self.assertEqual(output, ["a", "b", "c", "d", "e", "f", "g", "h"])

    def test_remove_words_with_duplicate_letters(self):
        test_cases = [
            (
                "CCRRIIAADDOO    MMEEDDIIAANNTTEE    OO    AARRTTIIGGOO",
                "CRIADO    MEDIANTE    O    ARTIGO",
            )
        ]
        for inputt, expected_output in test_cases:
            output = remove_word_with_duplicate_letters(inputt)
            self.assertEqual(output, expected_output)

    def test_remove_line_with_punctuation_only(self):
        test_cases = [
            (
                "deliberações. ............................................................................................... § 5.º",
                "deliberações. § 5.º",
            ),
            (
                "deliberações.     .$  %^&*()_+  \ \]['; /.,'] § 5.º",
                "deliberações. § 5.º",
            ),
            (
                "Procuradorias; ...........................................................................................” (NR)",
                "Procuradorias; (NR)",
            ),
            (
                "Art.34.................................................................................... III",
                "Art.34 III",
            ),
        ]
        for inputt, expected_output in test_cases:
            output = remove_line_with_punctuation_only(inputt)
            self.assertEqual(output, expected_output)

    def test_remove_special_quotes(self):
        test_cases = [
            (
                "“deliberações.”",
                '"deliberações."',
            ),
        ]
        for inputt, expected_output in test_cases:
            output = remove_special_quotes(inputt)
            self.assertEqual(output, expected_output)

    def test_sentence_segmentation(self):
        test_cases = [
            (
                "Primeira frase do texto que precisa ser segmentado por sentenças. Segunda frase do texto que precisa ser segmentado por sentenças",
                [
                    "Primeira frase do texto que precisa ser segmentado por sentenças.",
                    "Segunda frase do texto que precisa ser segmentado por sentenças",
                ],
            ),
            (
                "Art. 4.º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei. Art. 5.º Compete à Secretaria Municipal de Meio Ambiente e Sustentabilidade (SEMMAS) a emissão da autorização para a execução de poda de árvores.",
                [
                    "Art. 4.º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                    "Art. 5.º Compete à Secretaria Municipal de Meio Ambiente e Sustentabilidade (SEMMAS) a emissão da autorização para a execução de poda de árvores.",
                ],
            ),
            (
                "O PREFEITO DE MANAUS, no uso da competência que lhe confere o art. 80, inc. XI, da Lei Orgânica do Município de Manaus, resolve NOMEAR, a contar de 01-06-2022, nos termos do art. 11, inc. II, da Lei nº 1.118, de 01-09-1971 – Estatuto dos Servidores Públicos do Município de Manaus, o senhor MARCOS BRANDÃO CUNHA FILHO para exercer o cargo de Assessor I, simbologia CAD-3, integrante da estrutura organizacional da AGÊNCIA REGULADORA DOS SERVIÇOS PÚBLICOS DELEGADOS DO MUNICÍPIO DE MANAUS – AGEMAN, objeto da Lei nº 2.265, de 11 de dezembro de 2017, combinada com a Lei nº 2.627, de 01-07-2020. ",
                [
                    "O PREFEITO DE MANAUS, no uso da competência que lhe confere o art. 80, inc.XI, da Lei Orgânica do Município de Manaus, resolve NOMEAR, a contar de 01-06-2022, nos termos do art. 11, inc.II, da Lei nº 1.118, de 01-09-1971 – Estatuto dos Servidores Públicos do Município de Manaus, o senhor MARCOS BRANDÃO CUNHA FILHO para exercer o cargo de Assessor I, simbologia CAD-3, integrante da estrutura organizacional da AGÊNCIA REGULADORA DOS SERVIÇOS PÚBLICOS DELEGADOS DO MUNICÍPIO DE MANAUS – AGEMAN, objeto da Lei nº 2.265, de 11 de dezembro de 2017, combinada com a Lei nº 2.627, de 01-07-2020.",
                ],
            ),
            (
                'Fica acrescido o inciso X ao art. 73 da Lei n. 605, de 24 de julho de 2001, com a seguinte redação: "Art. 73.  X – recursos provenientes de reposição florestal." (NR)',
                [
                    'Fica acrescido o inciso X ao art. 73 da Lei n. 605, de 24 de julho de 2001, com a seguinte redação: "Art. 73. X – recursos provenientes de reposição florestal." (NR)'
                ],
            ),
        ]
        for inputt, expected_output in test_cases:
            output = list(sentence_segmentation(inputt))
            self.assertEqual(output, expected_output)

    def test_filter_directory_by_date(self):
        directory_path = "a/b/c/2022-06-21"
        result = _is_directory_date(directory_path, "2022-06-19")
        self.assertTrue(result)
        result = _is_directory_date(directory_path, "2022-06-20")
        self.assertTrue(result)
        result = _is_directory_date(directory_path, "2022-06-21")
        self.assertTrue(result)
        result = _is_directory_date(directory_path, "2022-06-22")
        self.assertFalse(result)

    def test_remove_special_chars(self):
        test_cases = [
            (
                "Art. 4.º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                "Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
            ),
            (
                "Art. 40.º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                "Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
            ),
            (
                "Art.5.º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                "Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
            ),
            (
                "Art.5. Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                "Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
            ),
            (
                "Art.5º Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
                "Fica sob responsabilidade do detentor da autorização a destinação dos resíduos da poda de árvores para as centrais de recebimento referidas no inciso I do art. 3.º desta Lei.",
            ),
            (
                "§ 2.º A Comissão poderá reunir quantas vezes forem",
                "A Comissão poderá reunir quantas vezes forem",
            ),
            (
                "§ 2º A Comissão poderá reunir quantas vezes forem",
                "A Comissão poderá reunir quantas vezes forem",
            ),
            (
                "§ 2. A Comissão poderá reunir quantas vezes forem",
                "A Comissão poderá reunir quantas vezes forem",
            ),
            (
                "§ 2 A Comissão poderá reunir quantas vezes forem",
                "A Comissão poderá reunir quantas vezes forem",
            ),
            (
                "§ 20 A Comissão poderá reunir quantas vezes forem",
                "A Comissão poderá reunir quantas vezes forem",
            ),
            (
                """
LEI:


Art. 1.º Fica instituído, no município de Manaus, o Junho
Verde, mês dedicado à realização de ações voltadas à sensibilização  da
população sobre a importância de ações de sustentabilidade e de
conservação do meio ambiente.


Art. 2.º As ações alusivas ao mês Junho Verde, sempre
que possível, deverão incluir atividades representativas direcionadas às
datas já consagradas, sem prejuízo da inclusão de outros, dentre eles:

I – Dia Nacional da Educação Ambiental: 3 de junho;
II – Dia Mundial do Meio Ambiente e Dia da Ecologia: 5 de

junho;
III – Dia dos Catadores de Materiais Recicláveis: 7 de

junho;
IV – Dia do Combate à Desertificação e à Seca: 17 de

junho.

Art. 3.º O mês Junho Verde terá como principais objetivos:
I – promover o debate, a avaliação e a organização de

propostas para a política ambiental;
II – incentivar a educação ambiental, por intermédio da

realização de debates e discussões, desenvolvendo uma ação
educacional que sensibilize a sociedade quanto ao dever de defesa e
preservação do ambiente;

III – incentivar a participação de entidades civis
organizadas na formulação das propostas de políticas ambientais, por
meio de entidades de classe, organizações não governamentais,
conselhos municipais e estaduais, entre outros;

IV – promover o plantio de árvores; e
V – fomentar a criação de associações de conservação da

natureza.

Art. 4.º O Órgão Municipal de Meio Ambiente poderá

realizar a cada ano, a critério dos seus gestores, em cooperação com
entidades públicas, entidades civis e outras organizações profissionais e
científicas, campanhas visando a aumentar a sensibilização sobre a
importância de ações de sustentabilidade e conservação do meio
ambiente.


Art. 5.º São símbolos do mês Junho Verde a fita de cor

verde bem como o uso dessa tonalidade em recursos visuais de
impacto, como a iluminação noturna em locais onde se possa dar
visibilidade ao tema, dentre outros.

Art. 6.º Fica incluído, no Calendário Oficial da Cidade de
Manaus, o mês Junho Verde.


Art. 7.º Esta Lei entra em vigor na data de sua publicação.

""",
                """
LEI:


Fica instituído, no município de Manaus, o Junho
Verde, mês dedicado à realização de ações voltadas à sensibilização  da
população sobre a importância de ações de sustentabilidade e de
conservação do meio ambiente.


As ações alusivas ao mês Junho Verde, sempre
que possível, deverão incluir atividades representativas direcionadas às
datas já consagradas, sem prejuízo da inclusão de outros, dentre eles:

I – Dia Nacional da Educação Ambiental: 3 de junho;
II – Dia Mundial do Meio Ambiente e Dia da Ecologia: 5 de

junho;
III – Dia dos Catadores de Materiais Recicláveis: 7 de

junho;
IV – Dia do Combate à Desertificação e à Seca: 17 de

junho.

O mês Junho Verde terá como principais objetivos:
I – promover o debate, a avaliação e a organização de

propostas para a política ambiental;
II – incentivar a educação ambiental, por intermédio da

realização de debates e discussões, desenvolvendo uma ação
educacional que sensibilize a sociedade quanto ao dever de defesa e
preservação do ambiente;

III – incentivar a participação de entidades civis
organizadas na formulação das propostas de políticas ambientais, por
meio de entidades de classe, organizações não governamentais,
conselhos municipais e estaduais, entre outros;

IV – promover o plantio de árvores; e
V – fomentar a criação de associações de conservação da

natureza.

O Órgão Municipal de Meio Ambiente poderá

realizar a cada ano, a critério dos seus gestores, em cooperação com
entidades públicas, entidades civis e outras organizações profissionais e
científicas, campanhas visando a aumentar a sensibilização sobre a
importância de ações de sustentabilidade e conservação do meio
ambiente.


São símbolos do mês Junho Verde a fita de cor

verde bem como o uso dessa tonalidade em recursos visuais de
impacto, como a iluminação noturna em locais onde se possa dar
visibilidade ao tema, dentre outros.

Fica incluído, no Calendário Oficial da Cidade de
Manaus, o mês Junho Verde.


Esta Lei entra em vigor na data de sua publicação.

""",
            ),
        ]
        for inputt, expected_output in test_cases:
            output = remove_special_line_prefix(inputt)
            self.assertEqual(output, expected_output)


class PreprocessingFileTests(unittest.TestCase):
    def test_preprocessing_file(self):
        text_file = "tests/054847a8785e44e832b796ef21f5cd4619d32173.txt"
        destination_file = "tests/clean_054847a8785e44e832b796ef21f5cd4619d32173.txt"
        if os.path.exists(destination_file):
            os.remove(destination_file)
        preprocess_gazette_txt_file(text_file, destination_file)
        self.assertTrue(os.path.exists(destination_file))

    def test_sentence_file_creation(self):
        test_cases = [
            (
                "tests/054847a8785e44e832b796ef21f5cd4619d32173.txt",
                "tests/sentence_054847a8785e44e832b796ef21f5cd4619d32173.txt",
            ),
            (
                "tests/a8b95c77d984aa228dc8c9c6094b526857df6878.txt",
                "tests/sentence_a8b95c77d984aa228dc8c9c6094b526857df6878.txt",
            ),
            (
                "tests/c3789a9296485ff74a139114f5925e71c4f548d9.txt",
                "tests/sentence_c3789a9296485ff74a139114f5925e71c4f548d9.txt",
            ),
        ]
        for text_file, destination_file in test_cases:
            if os.path.exists(destination_file):
                os.remove(destination_file)
            create_sentence_file(text_file, destination_file)
            self.assertTrue(os.path.exists(destination_file))


class FileIterationTests(unittest.TestCase):
    def setUp(self):
        self.cities_ids = ["1234", "4321"]
        self.days = ["18", "19", "20", "21"]

    def create_file_gazette_files(self, root_dir: str):
        file_name = "a"
        all_files = []
        for city in self.cities_ids:
            for day in self.days:
                os.makedirs(f"{root_dir}/{city}/2022-06-{day}")
                with open(
                    f"{root_dir}/{city}/2022-06-{day}/{file_name}.txt", "w"
                ) as gazette:
                    gazette.write("{file_name}.txt")
                all_files.append(f"{root_dir}/{city}/2022-06-{day}/{file_name}.txt")
                file_name = chr(ord(file_name) + 1)
        return all_files

    def filter_file_date_since(self, files, date):
        DATE_FORMAT = "%Y-%m-%d"
        date = datetime.strptime(date, DATE_FORMAT)
        filter_files = []
        for filee in files:
            filee_date = filee.split("/")[-2]
            filee_date = datetime.strptime(filee_date, DATE_FORMAT)
            if filee_date >= date:
                filter_files.append(filee)
        return sorted(filter_files)

    def test_find_gazettes_files_by_date(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            all_files = self.create_file_gazette_files(tmpdirname)
            expected_files = self.filter_file_date_since(all_files, "2022-06-20")
            gazettes = sorted(list(find_gazette_files(tmpdirname, since="2022-06-20")))
            self.assertEquals(gazettes, expected_files)

            expected_files = self.filter_file_date_since(all_files, "2022-06-19")
            gazettes = sorted(list(find_gazette_files(tmpdirname, since="2022-06-19")))
            self.assertEquals(gazettes, expected_files)

    def test_find_all_gazettes_files(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            all_files = self.create_file_gazette_files(tmpdirname)
            gazettes = list(find_gazette_files(tmpdirname))
            self.assertEquals(gazettes, all_files)
