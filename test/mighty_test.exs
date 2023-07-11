defmodule MightyTest do
  use ExUnit.Case
  doctest Mighty

  test "greets the world" do
    assert Mighty.hello() == :world
  end
end
