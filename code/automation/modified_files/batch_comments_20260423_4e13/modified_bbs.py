def _draw_on_image_(self, image, bounding_box):
        # ...
        # Compute background corner coordinates
        coords_bg_tl, coords_bg_tr, coords_bg_bl, coords_bg_br = \
            self._compute_bg_corner_coords(self.image, bounding_box)
        
        # Draw background corner points
        for coords in [coords_bg_tl, coords_bg_tr, coords_bg_bl, coords_bg_br]:
            self.image.draw.draw_point(coords)
        
        # ...