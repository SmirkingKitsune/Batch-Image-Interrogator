"""Tests for the inquiry transcript viewer."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel

from ui.widgets import InquiryTranscriptWidget


def _turn(index: int, tag_count: int) -> dict:
    return {
        "prompt_type": "describe",
        "prompt_text": f"prompt {index} " + ("word " * 30),
        "included_tables": [],
        "included_transcripts": [],
        "sidecar_tags": [],
        "response_json": {
            "comment": f"[{index}] " + ("A sentence about the image. " * 30),
            "warnings": [],
        },
        "tags": [f"long_tag_name_{index}_{n}" for n in range(tag_count)],
        "model_name": "test-model",
    }


class TranscriptCardSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _transcript(self) -> InquiryTranscriptWidget:
        transcript = InquiryTranscriptWidget()
        transcript.resize(700, 900)
        transcript.show()
        self.app.processEvents()
        self.addCleanup(transcript.deleteLater)
        return transcript

    def test_card_height_does_not_collapse_as_tag_count_grows(self):
        """A tag-heavy turn must not shrink its row to a clipped sliver.

        Regression: the tag chips lived in an unbounded row, so a card with
        many tags had a natural width of thousands of pixels. Qt measured the
        card's sizeHint at that width, every word-wrapped label reported a
        single line, and the row ended up ~200px tall with the response and
        image clipped out of view.
        """
        transcript = self._transcript()
        for index, tag_count in enumerate((3, 60, 200)):
            transcript.append_turn_card(_turn(index, tag_count))
        self.app.processEvents()

        heights = [transcript.item(row).sizeHint().height() for row in range(3)]
        self.assertEqual(len(set(heights)), 1, f"row heights diverged: {heights}")
        self.assertGreater(heights[0], 250)

    def test_card_width_is_independent_of_tag_count(self):
        transcript = self._transcript()
        transcript.append_turn_card(_turn(0, 2))
        transcript.append_turn_card(_turn(1, 300))
        self.app.processEvents()

        widths = [
            transcript.itemWidget(transcript.item(row)).minimumSizeHint().width()
            for row in range(2)
        ]
        self.assertEqual(widths[0], widths[1])


class TranscriptPendingCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _transcript(self) -> InquiryTranscriptWidget:
        transcript = InquiryTranscriptWidget()
        transcript.resize(700, 900)
        transcript.show()
        self.app.processEvents()
        self.addCleanup(transcript.deleteLater)
        return transcript

    def test_pending_card_appears_before_any_response(self):
        transcript = self._transcript()
        handle = transcript.begin_turn_card(_turn(0, 0))

        self.assertEqual(transcript.count(), 1)
        self.assertTrue(handle.is_active)

    def test_streaming_text_grows_the_pending_row(self):
        transcript = self._transcript()
        handle = transcript.begin_turn_card(_turn(0, 0))
        self.app.processEvents()
        before = transcript.item(0).sizeHint().height()

        handle.set_stream_text("A streamed sentence about the image. " * 30)
        self.app.processEvents()

        self.assertGreater(transcript.item(0).sizeHint().height(), before)

    def test_complete_replaces_pending_card_in_place(self):
        transcript = self._transcript()
        handle = transcript.begin_turn_card(_turn(0, 0))
        handle.complete(_turn(0, 80))
        self.app.processEvents()

        self.assertEqual(transcript.count(), 1)
        self.assertFalse(handle.is_active)
        self.assertGreater(transcript.item(0).sizeHint().height(), 250)

    def test_handle_tolerates_a_cleared_transcript(self):
        """Switching image or mode clears the list while a request is in flight."""
        transcript = self._transcript()
        handle = transcript.begin_turn_card(_turn(0, 0))
        transcript.clear()

        self.assertFalse(handle.is_active)
        handle.set_stream_text("late chunk")
        handle.set_status("late status")
        handle.complete(_turn(0, 3))
        self.assertEqual(transcript.count(), 0)

    def test_fail_shows_the_error_and_keeps_the_request_visible(self):
        transcript = self._transcript()
        handle = transcript.begin_turn_card(_turn(0, 0))
        handle.fail("llama-server request failed")
        self.app.processEvents()

        card = transcript.itemWidget(transcript.item(0))
        texts = [label.text() for label in card.findChildren(QLabel)]
        self.assertTrue(any("llama-server request failed" in text for text in texts))

    def test_follow_tail_only_scrolls_when_parked_at_the_end(self):
        transcript = self._transcript()
        for index in range(8):
            transcript.append_turn_card(_turn(index, 4))
        self.app.processEvents()
        transcript.scrollToTop()
        self.app.processEvents()
        top = transcript.verticalScrollBar().value()

        transcript.follow_tail()

        self.assertEqual(transcript.verticalScrollBar().value(), top)


if __name__ == "__main__":
    unittest.main()
