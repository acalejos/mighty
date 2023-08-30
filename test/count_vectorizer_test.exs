defmodule CountVectorizerTest do
  use ExUnit.Case
  doctest Mighty.Preprocessing.CountVectorizer
  import Nx, only: :sigils

  setup do
    %{
      corpus: [
        "This is the first document",
        "This document is the second document",
        "And this is the third one",
        "Is this the first document"
      ]
    }
  end

  test "builds vocab", context do
    vectorizer = Mighty.Preprocessing.CountVectorizer.new(context[:corpus]) |> IO.inspect()

    vocab_keys = vectorizer.vocabulary |> Map.keys() |> MapSet.new()
    {tf, df} = Mighty.Preprocessing.CountVectorizer.transform(vectorizer, context[:corpus])
    expected_x = ~M<
    0 1 1 1 0 0 1 0 1
    0 2 0 1 0 1 1 0 1
    1 0 0 1 1 0 1 1 1
    0 1 1 1 0 0 1 0 1
    >

    assert tf == expected_x

    expected_keys =
      MapSet.new(["and", "document", "first", "is", "one", "second", "the", "third", "this"])

    assert vocab_keys == expected_keys

    expected_vocab = %{
      "this" => 8,
      "is" => 3,
      "the" => 6,
      "first" => 2,
      "document" => 1,
      "second" => 5,
      "and" => 0,
      "third" => 7,
      "one" => 4
    }

    assert vectorizer.vocabulary == expected_vocab
    assert vectorizer.fixed_vocabulary == false

    vectorizer = Mighty.Preprocessing.CountVectorizer.new(context[:corpus], ngram_range: {2, 2})
    {tf, df} = Mighty.Preprocessing.CountVectorizer.transform(vectorizer, context[:corpus])
    expected_x = ~M<
    0 0 1 1 0 0 1 0 0 0 0 1 0
    0 1 0 1 0 1 0 1 0 0 1 0 0
    1 0 0 1 0 0 0 0 1 1 0 1 0
    0 0 1 0 1 0 1 0 0 0 0 0 1
    >

    expected_x =
      expected_vocab =
      MapSet.new([
        "and this",
        "document is",
        "first document",
        "is the",
        "is this",
        "second document",
        "the first",
        "the second",
        "the third",
        "third one",
        "this document",
        "this is",
        "this the"
      ])

    vocab_keys = vectorizer.vocabulary |> Map.keys() |> MapSet.new()
    assert vocab_keys == expected_vocab

    expected_vocab = %{
      "this is" => 11,
      "is the" => 3,
      "the first" => 6,
      "first document" => 2,
      "this document" => 10,
      "document is" => 1,
      "the second" => 7,
      "second document" => 5,
      "and this" => 0,
      "the third" => 8,
      "third one" => 9,
      "is this" => 4,
      "this the" => 12
    }

    assert vectorizer.vocabulary == expected_vocab
  end
end
