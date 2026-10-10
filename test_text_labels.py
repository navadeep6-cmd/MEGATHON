import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest import labels, textutil  # noqa: E402


class TextUtilTests(unittest.TestCase):
    def test_strip_html(self):
        self.assertEqual(textutil.strip_html("<p>Eiffel&nbsp;Tower <b>at</b> night</p>"),
                         "Eiffel Tower at night")
        self.assertEqual(textutil.strip_html(None), "")

    def test_clean_names(self):
        self.assertEqual(textutil.clean_category_name("Category:Bridges_in_Paris"), "Bridges in Paris")
        self.assertEqual(textutil.clean_file_title("File:IMG_2041 Eiffel_Tower-at night.jpg"),
                         "Eiffel Tower at night")
        self.assertEqual(textutil.clean_file_title("File:DSC09876.JPG"), "")
        self.assertEqual(textutil.clean_file_title("File:PXL_20230101_123456789.jpg"), "")

    def test_tokens_and_stemming(self):
        self.assertEqual(textutil.light_stem("bridges"), "bridge")
        self.assertEqual(textutil.light_stem("churches"), "church")
        self.assertEqual(textutil.light_stem("cities"), "city")
        self.assertEqual(textutil.light_stem("glass"), "glass")
        toks = textutil.content_tokens("Churches of São Paulo, 2019")
        self.assertEqual(toks, frozenset({"church", "sao", "paulo", "2019"}))
        self.assertEqual(textutil.category_tokens("Views"), frozenset({"view"}))

    def test_accent_folding_keeps_other_scripts(self):
        self.assertEqual(textutil.fold_accents("Bogotá"), "Bogota")
        hindi = "दिवाली"
        self.assertEqual(textutil.fold_accents(hindi), hindi)

    def test_years(self):
        self.assertEqual(textutil.find_year("2019 in Paris"), 2019)
        self.assertIsNone(textutil.find_year("IMG 123456"))
        self.assertEqual(textutil.meta_year('<time datetime="2018-05-04">4 May 2018</time>'), 2018)
        self.assertIsNone(textutil.meta_year(""))

    def test_script(self):
        self.assertEqual(textutil.name_script("Bridges in Paris"), "ascii")
        self.assertEqual(textutil.name_script("Église Saint-Étienne"), "latin-extended")
        self.assertEqual(textutil.name_script("दिवाली"), "non-latin")

    def test_meta_text(self):
        self.assertEqual(textutil.build_meta_text("Eiffel Tower", "Seen from the Seine"),
                         "Eiffel Tower. Seen from the Seine")
        self.assertEqual(textutil.build_meta_text("", ""), "")


class LabelTests(unittest.TestCase):
    MAINTENANCE = [
        "CC-BY-SA-4.0", "GFDL", "PD-self", "Self-published work", "Files by User:Example",
        "Files from Flickr", "Images from Geograph Britain and Ireland",
        "Photographs taken on 2019-05-04", "Taken with Canon EOS 5D Mark III",
        "Uploaded with UploadWizard", "Media needing categories", "Quality images",
        "Featured pictures on Wikimedia Commons", "Pictures of the day (2020)",
        "Uncategorized images", "Wiki Loves Monuments 2023 in India",
        "Media missing infobox template", "Pages with maps", "Category:Images with annotations",
        "Photos by Example Person", "Deletion requests March 2024", "License migration redundant",
    ]
    CONTENT = [
        "Bridges in Paris", "Photographs of bridges by night", "Diwali 2019 in Delhi",
        "Black and white photographs of cats", "Paintings by Vincent van Gogh", "Licensed taxis",
        "User interfaces", "Steam locomotives of India", "Lighthouses in Kerala", "Sunsets",
        "Files (furniture)",
    ]

    def test_maintenance(self):
        for name in self.MAINTENANCE:
            self.assertTrue(labels.is_maintenance(name), name)

    def test_content(self):
        for name in self.CONTENT:
            self.assertFalse(labels.is_maintenance(name), name)

    def test_clean_labels(self):
        out = labels.clean_labels(
            ["Category:Bridges_in_Paris", "Bridges in Paris", "Old name", "GFDL", "Dropped"],
            redirects={"Old name": "New name"}, drop={"Dropped"})
        self.assertEqual(out, ["Bridges in Paris", "New name"])

    def test_redirect_cycle_terminates(self):
        out = labels.clean_labels(["A"], redirects={"A": "B", "B": "A"})
        self.assertEqual(len(out), 1)

    def test_licence(self):
        for ok in ("CC BY-SA 4.0", "CC0", "Public domain", "CC BY 2.0", "GFDL", "PD-old"):
            self.assertTrue(labels.license_ok(ok), ok)
        for bad in ("", None, "Fair use", "All rights reserved"):
            self.assertFalse(labels.license_ok(bad), bad)

    def test_person(self):
        self.assertTrue(labels.is_person_category(["Category:Men by name", "Politicians of India"]))
        self.assertTrue(labels.is_person_category(["People by name"]))
        self.assertFalse(labels.is_person_category(["Bridges in Paris"]))

    def test_types(self):
        self.assertEqual(labels.category_type("Diwali 2019 in Delhi"), "event")
        self.assertEqual(labels.category_type("Protests in Chile"), "event")
        self.assertEqual(labels.category_type("Reflections in water"), "abstract")
        self.assertEqual(labels.category_type("Beaches of Goa"), "scene")
        self.assertEqual(labels.category_type("Bridges in Paris"), "object")
        self.assertEqual(labels.category_type("Coffee cups"), "object")

    def test_bands(self):
        self.assertEqual(labels.frequency_band(0), "zero-shot")
        self.assertEqual(labels.frequency_band(3), "tail")
        self.assertEqual(labels.frequency_band(10), "torso")
        self.assertEqual(labels.frequency_band(50), "head")
        self.assertEqual(labels.resolution_band(640, 480), "<800px")
        self.assertEqual(labels.resolution_band(4000, 3000), ">=4000px")
        self.assertEqual(labels.resolution_band(None, 10), "unknown")


if __name__ == "__main__":
    unittest.main()
