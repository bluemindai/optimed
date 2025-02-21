import unittest
import numpy as np
from optimed.processes.postprocess import (
    if_touches_border_3d,
    if_touches_border_2d,
    find_component_by_centroid,
    find_component_by_point,
    delete_small_segments,
    delete_segments_disconnected_from_point,
    delete_segments_disconnected_from_parent,
    delete_segments_distant_from_point,
    delete_nearby_segments_with_buffer,
    delete_touching_border_segments,
    fill_holes_2d,
    fill_holes_3d
)

import warnings
warnings.filterwarnings("ignore")

class TestPostprocessFunctions(unittest.TestCase):

    def test_if_touches_border_3d(self):
        # Тест 1: Маска касается границы
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 2, 2] = 1  # На границе
        self.assertTrue(if_touches_border_3d(mask, use_gpu=False))

        # Тест 2: Маска не касается границы
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = 1  # В центре
        self.assertFalse(if_touches_border_3d(mask, use_gpu=False))

    def test_if_touches_border_2d(self):
        # Тест 1: Маска касается границы
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 2] = 1  # На границе
        self.assertTrue(if_touches_border_2d(mask, use_gpu=False))

        # Тест 2: Маска не касается границы
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = 1  # В центре
        self.assertFalse(if_touches_border_2d(mask, use_gpu=False))

    def test_find_component_by_centroid(self):
        # Маска с двумя компонентами
        mask = np.array([[0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 2, 2]])
        labeled = np.where(mask > 0, mask, 0)
        num_components = 2

        # Тест 1: Поиск компоненты 1 (центроид около [0.5, 1.5])
        self.assertEqual(find_component_by_centroid(labeled, [0.5, 1.5], num_components, atol=1, use_gpu=False), 1)

        # Тест 2: Поиск компоненты 2 (центроид около [3, 2.5])
        self.assertEqual(find_component_by_centroid(labeled, [3, 2.5], num_components, atol=1, use_gpu=False), 2)

        # Тест 3: Нет подходящей компоненты
        self.assertIsNone(find_component_by_centroid(labeled, [0, 0], num_components, atol=1, use_gpu=False))

    def test_find_component_by_point(self):
        # Маска с двумя компонентами
        mask = np.array([[0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 2, 2]])
        labeled = np.where(mask > 0, mask, 0)

        # Тест 1: Точка внутри komponенты 1
        self.assertEqual(find_component_by_point(labeled, [0, 1], use_gpu=False), 1)

        # Тест 2: Точка внутри komponенты 2
        self.assertEqual(find_component_by_point(labeled, [3, 2], use_gpu=False), 2)

        # Тест 3: Точка вне всех компонент
        self.assertIsNone(find_component_by_point(labeled, [0, 0], use_gpu=False))

    def test_delete_small_segments(self):
        # Маска с маленькими и большими компонентами
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        expected = np.array([[1, 1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        
        # Удаляем компоненты размером меньше 3
        result = delete_small_segments(mask, interval=[3, np.inf], use_gpu=False, verbose=False)
        np.testing.assert_array_equal(result, expected)

    def test_delete_segments_disconnected_from_point(self):
        # Маска с несколькими компонентами
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        expected = np.array([[1, 1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        
        # Сохраняем компоненту с точкой [0, 0]
        result = delete_segments_disconnected_from_point(mask, [0, 0], size_threshold=2, use_gpu=False)
        np.testing.assert_array_equal(result, expected)

    def test_delete_segments_disconnected_from_parent(self):
        # Бинарная маска и родительская маска
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        parent = np.array([[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        expected = np.array([[1, 1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        
        result = delete_segments_disconnected_from_parent(mask, parent, size_threshold=2, use_gpu=False, verbose=False)
        np.testing.assert_array_equal(result, expected)

    def test_delete_segments_distant_from_point(self):
        # Маска с несколькими компонентами
        mask = np.array([[1, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        expected = np.array([[1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        
        # Удаляем компоненты дальше 1 от точки [0, 0]
        result = delete_segments_distant_from_point(mask, [0, 0], distance_threshold=1, size_threshold=1, use_gpu=False, verbose=False)
        np.testing.assert_array_equal(result, expected)

    def test_delete_nearby_segments_with_buffer(self):
        mask = np.array([[1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        parent = np.array([[1, 1, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        expected = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        
        result = delete_nearby_segments_with_buffer(mask, parent, distance_threshold=1, size_threshold=2, use_gpu=False, verbose=False)
        np.testing.assert_array_equal(result, expected)

    def test_delete_touching_border_segments(self):
        # Маска с компонентами на границе
        mask = np.array([[1, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        expected = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 0]])
        
        # Удаляем маленькие компоненты на границе (меньше 3)
        result = delete_touching_border_segments(mask, size_threshold=3, use_gpu=False, verbose=False)
        np.testing.assert_array_equal(result, expected)

    def test_fill_holes_2d(self):
        mask = np.array([[1, 1, 1, 1],
                        [1, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 1, 1]])
        expected = np.array([[1, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 1]])  # Hole size 4 > 3, should not be filled
        
        result = fill_holes_2d(mask, max_hole_size=3, use_gpu=False)
        np.testing.assert_array_equal(result, expected)

    def test_fill_holes_3d(self):
        # 3D маска с дыркой
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, :, :] = 1
        mask[2, :, :] = 1
        mask[1, 0, :] = 1
        mask[1, 2, :] = 1
        mask[1, 1, 0] = 1
        mask[1, 1, 2] = 1
        expected = np.ones((3, 3, 3), dtype=bool)
        
        # Заполняем дырки размером до 2
        result = fill_holes_3d(mask, max_hole_size=2, use_gpu=False)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()