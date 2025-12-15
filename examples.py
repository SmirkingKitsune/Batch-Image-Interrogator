#!/usr/bin/env python3
"""
Example: Programmatic Usage of Image Interrogator

This script demonstrates how to use the interrogator framework
without the GUI for automated batch processing.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core import InterrogationDatabase, FileManager, hash_image_content, get_image_metadata
from interrogators import WDInterrogator, CLIPInterrogator


def example_basic_interrogation():
    """Example: Basic image interrogation with WD Tagger."""
    print("=== Example 1: Basic Interrogation ===\n")
    
    # Initialize interrogator
    interrogator = WDInterrogator()
    interrogator.load_model(threshold=0.35, device='cuda')
    
    # Interrogate single image
    image_path = "path/to/your/image.jpg"
    
    try:
        results = interrogator.interrogate(image_path)
        
        print(f"Tags: {', '.join(results['tags'])}")
        print(f"\nTop 5 tags with confidence:")
        for tag in results['tags'][:5]:
            conf = results['confidence_scores'][tag]
            print(f"  {tag}: {conf:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        interrogator.unload_model()


def example_batch_with_database():
    """Example: Batch processing with database caching."""
    print("\n=== Example 2: Batch Processing with Cache ===\n")
    
    # Setup
    db = InterrogationDatabase()
    interrogator = WDInterrogator()
    interrogator.load_model(threshold=0.35, device='cuda')
    
    # Register model once
    model_id = db.register_model(
        interrogator.model_name,
        interrogator.get_model_type(),
        config=interrogator.get_config()
    )
    
    # Get images
    directory = "path/to/image/directory"
    images = FileManager.find_images(directory)
    
    print(f"Processing {len(images)} images...\n")
    
    for idx, image_path in enumerate(images, 1):
        try:
            # Hash image
            file_hash = hash_image_content(str(image_path))
            
            # Check cache
            cached = db.get_interrogation(file_hash, interrogator.model_name)
            
            if cached:
                print(f"[{idx}/{len(images)}] {image_path.name} - Using cached results")
                results = cached
            else:
                print(f"[{idx}/{len(images)}] {image_path.name} - Interrogating...")
                results = interrogator.interrogate(str(image_path))
                
                # Save to database
                metadata = get_image_metadata(str(image_path))
                image_id = db.register_image(
                    str(image_path),
                    file_hash,
                    metadata['width'],
                    metadata['height'],
                    metadata['file_size']
                )
                
                db.save_interrogation(
                    image_id,
                    model_id,
                    results['tags'],
                    results['confidence_scores'],
                    results['raw_output']
                )
            
            # Write to file
            FileManager.write_tags_to_file(image_path, results['tags'])
            
        except Exception as e:
            print(f"[{idx}/{len(images)}] {image_path.name} - Error: {e}")
    
    interrogator.unload_model()
    db.close()
    print("\nBatch processing complete!")


def example_clip_interrogation():
    """Example: Using CLIP for natural language descriptions."""
    print("\n=== Example 3: CLIP Natural Language Descriptions ===\n")
    
    # Initialize CLIP
    interrogator = CLIPInterrogator()
    interrogator.load_model(mode='best', device='cuda')
    
    image_path = "path/to/your/image.jpg"
    
    try:
        results = interrogator.interrogate(image_path)
        
        print(f"Description: {results['raw_output']}")
        print(f"\nTags: {', '.join(results['tags'])}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        interrogator.unload_model()


def example_organize_by_tags():
    """Example: Organize images into subdirectories by tags."""
    print("\n=== Example 4: Organize by Tags ===\n")
    
    directory = "path/to/image/directory"
    images = FileManager.find_images(directory)
    
    # Get all available tags
    all_tags = FileManager.get_all_tags_in_directory(directory)
    print(f"Available tags: {', '.join(sorted(all_tags))}\n")
    
    # Organize portraits
    portrait_tags = ['portrait', 'face', 'person']
    moved_count = 0
    
    for image_path in images:
        was_moved = FileManager.organize_by_tags(
            image_path,
            portrait_tags,
            target_subdir='portraits',
            move_text=True,
            match_mode='any'
        )
        
        if was_moved:
            moved_count += 1
            print(f"Moved: {image_path.name}")
    
    print(f"\nMoved {moved_count} portrait images to ./portraits/")


def example_filter_by_confidence():
    """Example: Filter tags by confidence threshold."""
    print("\n=== Example 5: Filter Tags by Confidence ===\n")
    
    interrogator = WDInterrogator()
    interrogator.load_model(threshold=0.35, device='cuda')
    
    image_path = "path/to/your/image.jpg"
    
    try:
        # Get all results
        results = interrogator.interrogate(image_path)
        
        print("All tags (>0.35 confidence):")
        for tag in results['tags'][:10]:
            print(f"  {tag}: {results['confidence_scores'][tag]:.4f}")
        
        # Filter for high confidence only
        high_confidence_tags = [
            tag for tag, score in results['confidence_scores'].items()
            if score >= 0.6
        ]
        
        print(f"\nHigh confidence tags (>0.6):")
        for tag in high_confidence_tags:
            print(f"  {tag}: {results['confidence_scores'][tag]:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        interrogator.unload_model()


def example_compare_models():
    """Example: Compare CLIP and WD results on same image."""
    print("\n=== Example 6: Compare Models ===\n")
    
    image_path = "path/to/your/image.jpg"
    
    # WD Tagger
    print("WD Tagger Results:")
    wd = WDInterrogator()
    wd.load_model(threshold=0.35, device='cuda')
    wd_results = wd.interrogate(image_path)
    print(f"  Tags: {', '.join(wd_results['tags'][:10])}")
    wd.unload_model()
    
    # CLIP
    print("\nCLIP Results:")
    clip = CLIPInterrogator()
    clip.load_model(mode='fast', device='cuda')
    clip_results = clip.interrogate(image_path)
    print(f"  Description: {clip_results['raw_output']}")
    clip.unload_model()


if __name__ == '__main__':
    # Run examples
    print("Image Interrogator - Programmatic Usage Examples\n")
    print("=" * 60)
    
    # Uncomment the examples you want to run
    # Note: Update the image paths before running
    
    # example_basic_interrogation()
    # example_batch_with_database()
    # example_clip_interrogation()
    # example_organize_by_tags()
    # example_filter_by_confidence()
    # example_compare_models()
    
    print("\nUpdate image paths in the script and uncomment examples to run.")
