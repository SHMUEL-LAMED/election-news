import unittest
import local_blur as b


def person(box, score):
    return {'box': box, 'score': score}


class LocalBlurDecisionTests(unittest.TestCase):
    def test_only_a_consistently_male_track_stays_visible(self):
        man, woman = (10, 10, 60, 160), (200, 10, 250, 160)
        frames = [{'scene': 0, 'people': [person(man, 4.0), person(woman, -3.0)]} for _ in range(5)]
        self.assertEqual(b.decide(frames), [[woman]] * 5)

    def test_doubt_blurs(self):
        box = (10, 10, 60, 160)
        # Weak evidence, unjudged looks, and a single look are all treated as doubt.
        self.assertEqual(b.decide([{'scene': 0, 'people': [person(box, 1.0)]} for _ in range(6)]), [[box]] * 6)
        self.assertEqual(b.decide([{'scene': 0, 'people': [person(box, None)]} for _ in range(6)]), [[box]] * 6)
        self.assertEqual(b.decide([{'scene': 0, 'people': [person(box, 2.6)]}]), [[box]])
        self.assertEqual(b.decide([{'scene': 0, 'people': [person(box, 5.0)]}]), [[]])

    def test_female_looks_inside_a_male_average_blur_the_whole_track(self):
        box = (10, 10, 60, 160)
        scores = [5, 5, 5, 5, -2, -2, 5, 5]
        frames = [{'scene': 0, 'people': [person(box, s)]} for s in scores]
        self.assertTrue(sum(scores) / len(scores) > b.CLEAR_SCORE)
        self.assertEqual(b.decide(frames), [[box]] * len(scores))

    def test_one_extreme_look_cannot_clear_a_track(self):
        box = (10, 10, 60, 160)
        frames = [{'scene': 0, 'people': [person(box, s)]} for s in (40, 0.5, 0.5, 0.5)]
        self.assertEqual(b.decide(frames), [[box]] * 4)

    def test_tracks_follow_movement_and_stop_at_scene_cuts(self):
        frames = [{'scene': 0, 'people': [person((10 + 8 * i, 10, 60 + 8 * i, 160), 4.0)]} for i in range(4)]
        self.assertEqual(len(b.link_tracks(frames)), 1)
        # Same place after a cut is a different person until proven otherwise.
        frames.append({'scene': 1, 'people': [person((34, 10, 84, 160), -4.0)]})
        self.assertEqual(len(b.link_tracks(frames)), 2)
        self.assertEqual(b.decide(frames), [[], [], [], [], [(34, 10, 84, 160)]])

    def test_policy_boxes_are_valid_and_cover_the_figure(self):
        self.assertEqual(b.to_policy_boxes([(64, 36, 320, 180)], 640, 360), [[100, 100, 500, 500]])
        self.assertEqual(b.to_policy_boxes([(-5, -5, 700, 400)], 640, 360), [[0, 0, 1000, 1000]])
        self.assertEqual(b.to_policy_boxes([(5, 5, 5.2, 9)], 640, 360), [])
        crowd = [(i, 0, i + 5, 20) for i in range(0, 600, 10)]
        self.assertEqual(b.to_policy_boxes(crowd, 640, 360), [[0, 0, 1000, 1000]])


if __name__ == '__main__':
    unittest.main()
