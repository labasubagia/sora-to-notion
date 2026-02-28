"""
Unit tests for models.py - Pydantic models validation.
"""

import pytest
from pydantic import ValidationError

from models import ChatGPTImageGeneration, SoraImageGeneration


class TestChatGPTImageGeneration:
    """Tests for ChatGPTImageGeneration model."""

    def test_valid_model(self):
        """Should create model with valid data."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
        }
        model = ChatGPTImageGeneration(**data)
        assert model.id == "gen_123"
        assert model.prompt == "A test prompt"

    def test_missing_required_field(self):
        """Should raise ValidationError for missing required fields."""
        data = {
            "id": "gen_123",
            # Missing other required fields
        }
        with pytest.raises(ValidationError):
            ChatGPTImageGeneration(**data)

    def test_extra_fields_forbidden(self):
        """Should raise ValidationError for extra fields."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
            "extra_field": "not allowed",
        }
        with pytest.raises(ValidationError, match="extra_field"):
            ChatGPTImageGeneration(**data)

    def test_empty_prompt_allowed(self):
        """Empty prompt should be allowed (has default)."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
        }
        model = ChatGPTImageGeneration(**data)
        assert model.prompt == ""

    def test_wrong_type(self):
        """Should raise ValidationError for wrong types."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": 123,  # Should be string
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
        }
        with pytest.raises(ValidationError):
            ChatGPTImageGeneration(**data)


class TestSoraImageGeneration:
    """Tests for SoraImageGeneration model."""

    def test_valid_model(self):
        """Should create model with valid data."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_456",
            "task_id": "task_456",
            "url": "https://example.com/video.mp4",
            "prompt": "A test video prompt",
        }
        model = SoraImageGeneration(**data)
        assert model.id == "gen_456"
        assert model.task_id == "task_456"

    def test_minimal_valid_model(self):
        """Should create model with only required field (id)."""
        data = {"id": "gen_456"}
        model = SoraImageGeneration(**data)
        assert model.id == "gen_456"
        assert model.created_at is None
        assert model.task_id is None
        assert model.url is None
        assert model.prompt == ""

    def test_optional_fields_none(self):
        """Optional fields can be None."""
        data = {
            "created_at": None,
            "id": "gen_456",
            "task_id": None,
            "url": None,
            "prompt": "",
        }
        model = SoraImageGeneration(**data)
        assert model.created_at is None
        assert model.task_id is None
        assert model.url is None

    def test_missing_required_field(self):
        """Should raise ValidationError for missing id."""
        data = {"prompt": "test"}
        with pytest.raises(ValidationError):
            SoraImageGeneration(**data)

    def test_extra_fields_forbidden(self):
        """Should raise ValidationError for extra fields."""
        data = {
            "id": "gen_456",
            "extra_field": "not allowed",
        }
        with pytest.raises(ValidationError, match="extra_field"):
            SoraImageGeneration(**data)

    def test_wrong_type(self):
        """Should raise ValidationError for wrong types."""
        data = {
            "id": 456,  # Should be string
        }
        with pytest.raises(ValidationError):
            SoraImageGeneration(**data)

    def test_url_can_be_string_or_none(self):
        """URL can be string or None."""
        data1 = {"id": "gen_456", "url": "https://example.com/video.mp4"}
        model1 = SoraImageGeneration(**data1)
        assert model1.url == "https://example.com/video.mp4"

        data2 = {"id": "gen_456", "url": None}
        model2 = SoraImageGeneration(**data2)
        assert model2.url is None


class TestImageGenerationProtocol:
    """Tests for ImageGeneration protocol compatibility."""

    def test_chatgpt_has_required_attributes(self):
        """ChatGPT generation should have id and prompt."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "Test",
        }
        gen = ChatGPTImageGeneration(**data)
        assert hasattr(gen, "id")
        assert hasattr(gen, "prompt")
        assert gen.id == "gen_123"
        assert gen.prompt == "Test"

    def test_sora_has_required_attributes(self):
        """Sora generation should have id and prompt."""
        data = {"id": "gen_456", "prompt": "Test video"}
        gen = SoraImageGeneration(**data)
        assert hasattr(gen, "id")
        assert hasattr(gen, "prompt")
        assert gen.id == "gen_456"
        assert gen.prompt == "Test video"

    def test_both_models_compatible(self):
        """Both models should have same protocol attributes."""
        chatgpt = ChatGPTImageGeneration(
            created_at="2024-01-15T10:30:00.000000+00:00",
            id="gen_1",
            conversation_id="conv",
            message_id="msg",
            asset_pointer="asset",
            url="http://example.com",
            prompt="test",
        )
        sora = SoraImageGeneration(id="gen_2", prompt="test video")

        # Both should have id and prompt
        for gen in [chatgpt, sora]:
            assert hasattr(gen, "id")
            assert hasattr(gen, "prompt")
            assert isinstance(gen.id, str)
            assert isinstance(gen.prompt, str)
